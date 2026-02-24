import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from collections import Counter


def plot_segmented_deltas(
    timestamp_deltas: pd.Series, segments_dict: Dict[Tuple[int, int], Tuple]
):
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (k, v) in enumerate(segments_dict.items()):
        start, end = k
        color = v
        segment_deltas = timestamp_deltas.iloc[start : end + 1]
        ax.plot(
            segment_deltas.index,
            segment_deltas.values,
            marker="o",
            color=color,
            alpha=0.6,
            label=f"Segment {i + 1}",
        )

    ax.set_xlabel("Index")
    ax.set_ylabel("Δt (seconds)")
    ax.set_title("Time Differences per Segment")
    ax.legend(labelcolor="black", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    plt.show()


def extract_kblocks(trace_set: List[np.ndarray], k: int = 2):
    """
    Extract all overlapping k-blocks from a set of traces.
    Args:
        trace_set: np.ndarray of shape (num_traces, trace_length)
        k: Block size"""
    all_blocks = []
    for trace in trace_set:
        if trace.shape[0] < k:
            continue
        for i in range(len(trace) - k + 1):
            block = tuple(trace[i : i + k].astype(np.int32))
            all_blocks.append(block)
    return all_blocks


def calc_block_prob(trace_block_freq: dict, log_block_freq: dict):
    blocks_prob = {}

    for block, freq in trace_block_freq.items():
        if block in list(log_block_freq.keys()):
            prob = freq / log_block_freq[block]
            blocks_prob[block] = prob
        else:
            blocks_prob[block] = 0

    return blocks_prob


def k_block_likelihood_estimator(trace, L, k):
    """
    Compute the k-block likelihood estimator p̂_L for a trace.

    Parameters:
    -----------
    trace : list or array-like
        The observed trace/log of events
    L : set or list
        The log L (set of observed sequences/patterns)
    k : int
        Block length

    Returns:
    --------
    dict : mapping from k-blocks to their probability estimates
    """
    # Extract all k-blocks from the trace
    if len(trace) < k:
        return {}

    k_blocks = []
    for i in range(len(trace) - k + 1):
        block = tuple(trace[i : i + k])
        k_blocks.append(block)

    # Count occurrences of each k-block
    block_counts = Counter(k_blocks)

    # Compute f(L, k) - total number of k-blocks across the log
    f_L_k = 0
    for sigma in L:
        sigma_len = len(sigma)
        f_L_k += max(0, sigma_len - k + 1)

    # Handle the case where f(L, k) = 0
    if f_L_k == 0:
        return {block: 0 for block in block_counts.keys()}

    # Compute probability estimates
    p_estimates = {}
    for block in block_counts.keys():
        n_block = block_counts[block]
        p_estimates[block] = n_block / f_L_k

    return p_estimates


def k_block_entropy(trace, L, k):
    """
    Compute the k-block entropy estimator entropy^bl_k(L).

    Parameters:
    -----------
    trace : list or array-like
        The observed trace/log of events
    L : set or list
        The log L (set of observed sequences/patterns)
    k : int
        Block length

    Returns:
    --------
    float : the k-block entropy estimate
    """
    # Get probability estimates
    p_estimates = k_block_likelihood_estimator(trace, L, k)

    if not p_estimates:
        return 0.0

    # Compute entropy: -sum(p * log(p))
    entropy = 0.0
    for block, p in p_estimates.items():
        if p > 0:  # Avoid log(0)
            block_entropy = p * np.log2(p)
            entropy -= block_entropy  # Using log base 2
            print(block, block_entropy)

    return entropy
