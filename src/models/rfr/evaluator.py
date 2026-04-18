import numpy as np
import torch

from src.helpers.utils import _compute_window_fairness
from src.models.rfr.models import Adversary, NetRegression, NeuralGBDT
from src.models.rfr.rfr import RFR


def _build_rfr_model(data_dim, backbone='netregression', hidden_dim=50, n_ensemble=4):
    backbone = str(backbone).lower()
    if backbone == 'netregression':
        return NetRegression(data_dim, 1, size=int(hidden_dim)).float()
    if backbone == 'neuralgbdt':
        return NeuralGBDT(
            input_dim=int(data_dim),
            output_dim=1,
            n_ensemble=int(n_ensemble),
            hidden_dim=int(hidden_dim),
        ).float()
    raise ValueError(
        f"Unsupported RFR backbone: {backbone}. Use 'netregression' or 'neuralgbdt'."
    )


def _dp_penalty(outputs, sens):
    mask0 = sens == 0
    mask1 = sens == 1
    if mask0.any() and mask1.any():
        return torch.abs(outputs[mask0].mean() - outputs[mask1].mean()), True
    return torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device), False


def _dp_smooth_surrogate(outputs, sens):
    mask0 = sens == 0
    mask1 = sens == 1
    if mask0.any() and mask1.any():
        return outputs[mask0].mean() + outputs[mask1].mean(), True
    if mask0.any():
        return outputs[mask0].mean(), False
    if mask1.any():
        return outputs[mask1].mean(), False
    return torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device), False


def _fcr_loss(model, x_batch, a_batch, threshold=0.8):
    pseudo_outputs = model(x_batch).flatten()
    pseudo_label = pseudo_outputs.detach()
    max_probs = torch.maximum(pseudo_label, 1.0 - pseudo_label)
    targets = (pseudo_label > 0.5).float()
    mask = (max_probs >= float(threshold)).float()

    x_trans = x_batch + 0.01 * x_batch[torch.randperm(x_batch.size(0))]
    logits = model(x_trans).flatten()

    loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    sub_losses = []
    sizes = []

    for group in (0, 1):
        for label in (0, 1):
            subgroup = (a_batch == group) & (targets == float(label))
            if subgroup.any():
                logits_sub = logits[subgroup]
                targets_sub = targets[subgroup]
                mask_sub = mask[subgroup]
                denom = torch.clamp(mask_sub.sum(), min=1.0)
                per_elem = torch.nn.functional.binary_cross_entropy(
                    logits_sub,
                    targets_sub,
                    reduction='none',
                )
                sub_loss = (per_elem * mask_sub).sum() / denom
                sizes.append(float(mask_sub.sum().item()))
            else:
                sub_loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
                sizes.append(0.0)
            sub_losses.append(sub_loss)

    if all(size > 0 for size in sizes):
        coeffs = [1.0 / size for size in sizes]
        coeff_sum = sum(coeffs)
        coeffs = [coeff / coeff_sum for coeff in coeffs]
        for coeff, sub_loss in zip(coeffs, sub_losses):
            loss = loss + (sub_loss * float(coeff))
        return loss

    denom = torch.clamp(mask.sum(), min=1.0)
    per_elem = torch.nn.functional.binary_cross_entropy(
        logits,
        targets,
        reduction='none',
    )
    return (per_elem * mask).sum() / denom


def _prepare_batch(x_buffer, y_buffer, a_buffer, train_batch_size):
    n_items = len(x_buffer)
    if n_items == 0:
        return None, None, None

    batch_size = min(int(train_batch_size), n_items)
    if batch_size == n_items:
        indices = np.arange(n_items)
    else:
        indices = np.random.choice(n_items, size=batch_size, replace=False)

    x_batch = torch.tensor(np.asarray([x_buffer[i] for i in indices], dtype=np.float32))
    y_batch = torch.tensor(np.asarray([y_buffer[i] for i in indices], dtype=np.float32))
    a_batch = torch.tensor(np.asarray([a_buffer[i] for i in indices], dtype=np.int64))
    return x_batch, y_batch, a_batch


def _apply_dnn_update(model, optimizer, criterion, x_batch, y_batch, a_batch, penalty_coefficient):
    optimizer.zero_grad()
    outputs = model(x_batch).flatten()
    loss = criterion(outputs, y_batch)
    dp_loss, has_both_groups = _dp_penalty(outputs, a_batch)
    if has_both_groups:
        loss = loss + (float(penalty_coefficient) * dp_loss)
    loss.backward()
    optimizer.step()


def _apply_rfr_update(model, optimizer, criterion, x_batch, y_batch, a_batch, penalty_coefficient):
    optimizer.zero_grad()
    outputs = model(x_batch).flatten()
    smooth_loss, _ = _dp_smooth_surrogate(outputs, a_batch)
    if smooth_loss.grad_fn is None:
        smooth_loss = criterion(outputs, y_batch)
    smooth_loss.backward()
    optimizer.first_step(zero_grad=True)

    second_outputs = model(x_batch).flatten()
    second_loss = criterion(second_outputs, y_batch)
    dp_loss, has_both_groups = _dp_penalty(second_outputs, a_batch)
    if has_both_groups:
        second_loss = second_loss + (float(penalty_coefficient) * dp_loss)
    second_loss.backward()
    optimizer.second_step(zero_grad=True)


def _apply_adv_update(
    model,
    adversary,
    clf_optimizer,
    adv_optimizer,
    criterion,
    x_batch,
    y_batch,
    a_batch,
    penalty_coefficient,
):
    adv_optimizer.zero_grad()
    with torch.no_grad():
        clf_outputs_detached = model(x_batch).flatten().unsqueeze(1)
    adv_pred = adversary(clf_outputs_detached).flatten()
    adv_loss = criterion(adv_pred, a_batch.float()) * float(penalty_coefficient)
    adv_loss.backward()
    adv_optimizer.step()

    clf_optimizer.zero_grad()
    clf_outputs = model(x_batch).flatten()

    for param in adversary.parameters():
        param.requires_grad_(False)
    adv_pred_for_clf = adversary(clf_outputs.unsqueeze(1)).flatten()
    fairness_term = criterion(adv_pred_for_clf, a_batch.float()) * float(penalty_coefficient)
    clf_loss = criterion(clf_outputs, y_batch) - fairness_term
    clf_loss.backward()
    clf_optimizer.step()
    for param in adversary.parameters():
        param.requires_grad_(True)


def _apply_fcr_update(
    model,
    optimizer,
    criterion,
    x_batch,
    y_batch,
    a_batch,
    penalty_coefficient,
    fcr_threshold,
):
    optimizer.zero_grad()
    outputs = model(x_batch).flatten()
    loss = criterion(outputs, y_batch)
    loss = loss + (float(penalty_coefficient) * _fcr_loss(model, x_batch, a_batch, threshold=fcr_threshold))
    loss.backward()
    optimizer.step()


def evaluate_rfr_over_timesteps(
    x_test,
    y_test,
    a_test,
    approach='rfr',
    backbone='netregression',
    hidden_dim=50,
    n_ensemble=4,
    learning_rate=1e-3,
    rho=1e-4,
    penalty_coefficient=1.0,
    fcr_threshold=0.8,
    train_batch_size=64,
    buffer_size=512,
    adv_hidden_dim=32,
    accuracy_window=200,
    test_then_train=True,
):
    """Run RFR-family methods in online prequential mode over a test stream."""
    if len(x_test) == 0:
        return {
            'accuracy': [],
            'dp': [],
            'eo': [],
            'n_samples': 0,
            'drifted_points': [],
        }

    x0 = np.asarray(x_test[0], dtype=np.float32)
    data_dim = int(x0.shape[0])

    model = _build_rfr_model(
        data_dim=data_dim,
        backbone=backbone,
        hidden_dim=hidden_dim,
        n_ensemble=n_ensemble,
    )

    approach = str(approach).lower()
    supported = {'baseline', 'rfr', 'dnn', 'dnn_adv', 'dnn_fcr'}
    if approach not in supported:
        raise ValueError(
            f"Unsupported RFR approach: {approach}. Use one of {sorted(supported)}."
        )

    criterion = torch.nn.BCELoss()
    if approach in {'baseline', 'rfr'}:
        optimizer = RFR(
            model.parameters(),
            torch.optim.Adam,
            rho=float(rho),
            lr=float(learning_rate),
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=0.01,
        )

    adversary = None
    adv_optimizer = None
    if approach == 'dnn_adv':
        adversary = Adversary(1, n_hidden=int(adv_hidden_dim)).float()
        adv_optimizer = torch.optim.Adam(adversary.parameters(), lr=float(learning_rate))

    accuracies = []
    dps = []
    eos = []
    y_preds_all = []
    y_true_all = []
    a_all = []
    correct_buffer = []
    use_rolling = bool(accuracy_window)
    x_buffer = []
    y_buffer = []
    a_buffer = []
    max_buffer_size = max(2, int(buffer_size))
    effective_train_batch_size = max(1, int(train_batch_size))

    n_samples = len(x_test)
    print(f"Running RFR ({approach}) prequentially on {n_samples} samples...")

    for i in range(n_samples):
        x_i = torch.tensor(np.asarray(x_test[i], dtype=np.float32)).view(1, -1)
        y_i = int(y_test[i])
        a_i = int(a_test[i])

        model.eval()
        with torch.no_grad():
            prob = model(x_i).flatten()
            pred = int((prob >= 0.5).item())

        y_preds_all.append(pred)
        y_true_all.append(y_i)
        a_all.append(a_i)

        correct_buffer.append(int(pred == y_i))
        if use_rolling and len(correct_buffer) > accuracy_window:
            correct_buffer.pop(0)
        accuracies.append(float(sum(correct_buffer)) / len(correct_buffer))

        dp_val, eo_val = _compute_window_fairness(
            y_preds_all=y_preds_all,
            y_true_all=y_true_all,
            a_all=a_all,
            fairness_start=0,
            fairness_window=200,
        )
        dps.append(float(dp_val))
        eos.append(float(eo_val))

        if test_then_train:
            x_buffer.append(np.asarray(x_test[i], dtype=np.float32))
            y_buffer.append(float(y_i))
            a_buffer.append(int(a_i))

            if len(x_buffer) > max_buffer_size:
                x_buffer.pop(0)
                y_buffer.pop(0)
                a_buffer.pop(0)

            x_batch, y_batch, a_batch = _prepare_batch(
                x_buffer,
                y_buffer,
                a_buffer,
                train_batch_size=effective_train_batch_size,
            )
            if x_batch is None:
                continue

            model.train()
            if approach == 'baseline':
                _apply_rfr_update(
                    model,
                    optimizer,
                    criterion,
                    x_batch,
                    y_batch,
                    a_batch,
                    penalty_coefficient=0.0,
                )
            elif approach == 'rfr':
                _apply_rfr_update(
                    model,
                    optimizer,
                    criterion,
                    x_batch,
                    y_batch,
                    a_batch,
                    penalty_coefficient=penalty_coefficient,
                )
            elif approach == 'dnn':
                _apply_dnn_update(
                    model,
                    optimizer,
                    criterion,
                    x_batch,
                    y_batch,
                    a_batch,
                    penalty_coefficient=penalty_coefficient,
                )
            elif approach == 'dnn_adv':
                _apply_adv_update(
                    model,
                    adversary,
                    optimizer,
                    adv_optimizer,
                    criterion,
                    x_batch,
                    y_batch,
                    a_batch,
                    penalty_coefficient=penalty_coefficient,
                )
            elif approach == 'dnn_fcr':
                _apply_fcr_update(
                    model,
                    optimizer,
                    criterion,
                    x_batch,
                    y_batch,
                    a_batch,
                    penalty_coefficient=penalty_coefficient,
                    fcr_threshold=fcr_threshold,
                )

    print(f"RFR ({approach}) evaluation complete.")
    return {
        'accuracy': accuracies,
        'dp': dps,
        'eo': eos,
        'n_samples': n_samples,
        'drifted_points': [],
    }
