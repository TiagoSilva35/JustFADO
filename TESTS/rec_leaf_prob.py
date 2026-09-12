import numpy as np
import tensorflow as tf
import time

from src.models.forest.fdt import FairDecisionTree

node_decisions = [0.8, 0.2, 0.0, 0.0, 0.0, 0.23, 0.77]

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

output = recursive_updater(node_decisions, np.ones(2**3), 3)
print("output:", output)

tree_depth = 15
num_samples = 10
tree = FairDecisionTree(
    data_dim=2,
    tree_depth=tree_depth,
    num_classes=2,
    compute_mode="default",
)
inputs = tf.random.normal([num_samples, 2], dtype=tf.float32, seed=2)
mask_matrix = tf.cast(tree.mask_matrix, tf.float32)


def node_decisions_for_sample(sample):
    """Get routing decisions for one arriving sample."""
    logits = (
        tf.matmul(sample, tree.weight) + tree.bias
    ) / tree.temperature
    return tree.activation(logits)


def run_online(updater_name):
    """Process one sample at a time and time only leaf probability updates."""
    first_sample = tf.expand_dims(inputs[0], axis=0)
    first_decisions = node_decisions_for_sample(first_sample)
    if updater_name == "recursive":
        recursive_updater_tf(first_decisions, tree_depth)
    else:
        mask_updater(first_decisions, mask_matrix)

    start = time.perf_counter()
    final_recursive = tf.zeros([1, tree.num_leaves], dtype=tf.float32)
    final_mask = tf.zeros([1, tree.num_leaves], dtype=tf.float32)

    for sample in tf.unstack(inputs, axis=0):
        sample = tf.expand_dims(sample, axis=0)
        decisions = node_decisions_for_sample(sample)

        if updater_name == "recursive":
            result = recursive_updater_tf(decisions, tree_depth)
            final_recursive = result
        else:
            result = mask_updater(decisions, mask_matrix)
            final_mask = result

    elapsed = time.perf_counter() - start
    return elapsed, final_recursive, final_mask


mask_time, _, mask_result = 0, 0, [0]#run_online("mask")
print(
    f"Mask leaf probabilities computed in {mask_time:.4f}s total, "
    f"{mask_time / num_samples * 1000:.3f} ms/sample"
)
recursive_time, recursive_result, _ = run_online("recursive")
print(
    f"Recursive leaf probabilities computed in {recursive_time:.4f}s total, "
    f"{recursive_time / num_samples * 1000:.3f} ms/sample"
)

np.testing.assert_allclose(
    recursive_result.numpy(),
    mask_result.numpy(),
    rtol=1e-5,
    atol=1e-6,
    err_msg="Recursive and mask leaf probabilities do not match.",
)
print("Recursive and mask leaf probabilities match for 1000 online samples.")
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
