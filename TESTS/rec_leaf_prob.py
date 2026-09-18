import numpy as np
import tensorflow as tf
import time

from src.models.forest.fdt import FairDecisionTree
from matplotlib import pyplot as plt

def recursive_updater(node_decisions, probs, depth):  
    if depth == 0:
        return probs
    probs = recursive_updater(node_decisions, probs, depth-1)
    decisions = node_decisions[2**(depth-1)-1:2**depth-1] # 1 : 3 ; 3: 7
    next_probs = [0.0] * (len(probs) * 2)
    for i, decision in enumerate(decisions):
        next_probs[2*i] = probs[i] * decision
        next_probs[2*i+1] = probs[i] * (1 - decision)
    
    return next_probs


def fdt_recursive_updater(node_decisions, probs, depth):
    """Copy FairDecisionTree._recursive_updater_ exactly."""
    if depth == 0:
        return probs
    probs = fdt_recursive_updater(node_decisions, probs, depth - 1)
    decisions = node_decisions[2**(depth - 1) - 1:2**depth - 1]
    next_probs = [None] * (len(probs) * 2)
    for i, decision in enumerate(decisions):
        next_probs[2 * i] = probs[i] * decision
        next_probs[2 * i + 1] = probs[i] * (1 - decision)
    return tf.stack(next_probs)


@tf.function
def recursive_updater_tf(node_decisions, tree_depth):
    """Batched TensorFlow version used for the optimized online benchmark."""
    batch_size = tf.shape(node_decisions)[0]
    probs = tf.ones([batch_size, 1], dtype=node_decisions.dtype)

    for level in range(tree_depth):
        start = 2**level - 1
        end = 2**(level + 1) - 1
        decisions = node_decisions[:, start:end]
        probs = tf.reshape(
            tf.stack(
                [probs * decisions, probs * (1.0 - decisions)],
                axis=-1,
            ),
            [batch_size, -1],
        )

    return probs


def mask_updater(node_decisions, mask_matrix):
    """Copy the mask-based leaf-probability calculation from FairDecisionTree."""
    y = tf.expand_dims(node_decisions, axis=2)
    y_repeated = tf.repeat(y, mask_matrix.shape[1], axis=2)
    z = tf.multiply(y_repeated, mask_matrix)

    ones_nodes = tf.nn.relu(-mask_matrix)
    mask = tf.ones_like(mask_matrix) - tf.math.abs(mask_matrix)
    probs = tf.nn.relu(z) + (ones_nodes - tf.nn.relu(-z)) + mask
    probs += 1e-8

    return tf.math.reduce_prod(probs, axis=1)


def node_decisions_for_sample(sample, tree):
    """Get routing decisions for one arriving sample."""
    logits = (
        tf.matmul(sample, tree.weight) + tree.bias
    ) / tree.temperature
    return tree.activation(logits)



def run_online(updater_name, depth, tree, inputs, mask_matrix):
    """Process one sample at a time and time only leaf probability updates."""
    first_sample = tf.expand_dims(inputs[0], axis=0)
    first_decisions = node_decisions_for_sample(first_sample, tree)
    if updater_name == "recursive":
        recursive_updater_tf(first_decisions, depth)
    else:
        mask_updater(first_decisions, mask_matrix)

    start = time.perf_counter()
    final_recursive = tf.zeros([1, tree.num_leaves], dtype=tf.float32)
    final_mask = tf.zeros([1, tree.num_leaves], dtype=tf.float32)

    for sample in tf.unstack(inputs, axis=0):
        sample = tf.expand_dims(sample, axis=0)
        decisions = node_decisions_for_sample(sample, tree)

        if updater_name == "recursive":
            result = recursive_updater_tf(decisions, depth)
            final_recursive = result
        else:
            result = mask_updater(decisions, mask_matrix)
            final_mask = result

    elapsed = time.perf_counter() - start
    return elapsed, final_recursive, final_mask


def run_tree_depth(depth):
    mask_time, _, mask_result = run_online("mask", depth, tree, inputs, mask_matrix)
    print(
        f"Mask leaf probabilities computed in {mask_time:.4f}s total, "
        f"{mask_time / num_samples * 1000:.3f} ms/sample"
    )
    recursive_time, recursive_result, _ = run_online("recursive", depth, tree, inputs, mask_matrix)
    print(
        f"Recursive leaf probabilities computed in {recursive_time:.4f}s total, "
        f"{recursive_time / num_samples * 1000:.3f} ms/sample"
    )

    np.testing.assert_allclose(
        recursive_result.numpy(),
        mask_result.numpy(),
        rtol=1e-6,
        atol=1e-8,
        err_msg="Recursive and mask leaf probabilities do not match.",
    )
    print(f"Recursive and mask leaf probabilities match for {num_samples} samples.")
    print(
        f"recursive: {recursive_time:.4f}s total, "
        f"{recursive_time / num_samples * 1000:.3f} ms/sample"
    )
    print(
        f"mask: {mask_time:.4f}s total, "
        f"{mask_time / num_samples * 1000:.3f} ms/sample"
    )
    winner = "recursive" if recursive_time < mask_time else "mask"
    print(f"Fastest implementation: {winner}")
    return recursive_time, mask_time, winner
    
tree_depths = [3, 4, 5, 6, 7, 8, 9, 10, 11]
results = []

for depth in tree_depths:
    num_samples = 1000
    tree = FairDecisionTree(
        data_dim=2,
        tree_depth=depth,
        num_classes=2,
        compute_mode="default",
    )
    inputs = tf.random.normal([num_samples, 2], dtype=tf.float32, seed=2)
    mask_matrix = tf.cast(tree.mask_matrix, tf.float32)
    print(f"\nRunning tests for tree depth {depth}...")
    recursive_time, mask_time, winner = run_tree_depth(depth)
    results.append((depth, recursive_time, mask_time, winner))
    

for depth, recursive_time, mask_time, winner in results:
    print(f"Tree depth {depth}: Recursive={recursive_time:.4f}s, Mask={mask_time:.4f}s, Winner={winner}")
    
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

ax.plot(
    tree_depths,
    [r[1] for r in results],
    label="Recursive update",
    color="#0072B2",
    marker="o",
    markersize=6,
    linewidth=2.2,
)
ax.plot(
    tree_depths,
    [r[2] for r in results],
    label="Mask update",
    color="#D55E00",
    marker="s",
    markersize=5.5,
    linewidth=2.2,
)

ax.set_title("Leaf Probability Update Performance", fontsize=15, pad=12, weight="bold")
ax.set_xlabel("Tree depth", fontsize=11)
ax.set_ylabel("Elapsed time (seconds)", fontsize=11)
ax.set_xticks(tree_depths)
ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
plt.show()