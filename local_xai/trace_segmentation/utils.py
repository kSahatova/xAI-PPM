import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def build_transition_probability_matrix(log_df: pd.DataFrame, unique_activities_num: int = 25, special_tokens_num: int = 3):
    n_cols = n_rows = special_tokens_num + unique_activities_num
    adjacency = np.zeros((n_rows, n_cols), dtype=int)

    for case_id, group in log_df.groupby('case_id'):
        activities = group['activity'].values
        for i in range(len(activities) - 1):
            adjacency[activities[i], activities[i + 1]] += 1

    transition_matrix = adjacency / adjacency.sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix).astype(np.float64)

    return transition_matrix


def plot_trace_vertical(
    trace: np.ndarray,
    seg_boundaries_colors: Dict[Tuple[int, int], Tuple],
    activity_lookup: Dict[int, str],
    figsize: Tuple[int, int] = (8, 12),
    cell_height: float = 1,
    show_indices: bool = True,
    title: str = "Trace Segmentation",
):
    """
    Visualize trace activities vertically with colored segments.

    Args:
        trace: numpy array of shape (sequence_length, num_features)
        seg_boundaries_colors: Dictionary mapping (start, end) tuples to RGB color tuples
        activity_lookup: Dictionary mapping activity tokens to activity names
        activity_token_index: Index of the activity token in trace features
        figsize: Figure size (width, height)
        cell_height: Height of each activity cell
        show_indices: Whether to show event indices
        title: Plot title

    Returns:
        fig: matplotlib figure object
    """
    # Extract activity tokens
    activity_tokens = trace.astype(int)
    trace_length = len(activity_tokens)

    # Create segment membership mapping
    segment_membership = {}
    for (start, end), color in seg_boundaries_colors.items():
        for i in range(start, end + 1):
            if i < trace_length:
                segment_membership[i] = color

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # Plot each activity as a horizontal bar
    for i, token in enumerate(activity_tokens):
        y_pos = trace_length - i - 1  # Reverse so first event is at top

        # Get color for this position
        color = segment_membership.get(i, (0.85, 0.85, 0.85))  # Default gray

        # Draw colored rectangle
        rect = patches.Rectangle(
            (0, y_pos),
            1,
            cell_height,
            linewidth=0,
            edgecolor="none",
            facecolor=color,
            alpha=0.6,
        )
        ax.add_patch(rect)

        # Add activity name
        activity_name = activity_lookup.get(token, f"Activity_{token}")
        # fontsize = min(*figsize) * 10
        ax.text(
            0.5,
            y_pos + cell_height / 2,
            activity_name,
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="black",
        )

        # Add index on the left
        if show_indices:
            ax.text(
                -0.15,
                y_pos + cell_height / 2,
                f"{i}",
                ha="right",
                va="center",
                fontsize=8,
                color="gray",
            )

    # Set limits and labels
    ax.set_xlim(-0.2 if show_indices else 0, 1)
    ax.set_ylim(0, trace_length)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add segment labels on the right
    for seg_num, ((start, end), color) in enumerate(seg_boundaries_colors.items()):
        # if end < trace_length:
        mid_y = trace_length - (start + end) / 2 - 1
        segment_length = end - start + 1
        ax.text(
            1.05,
            mid_y,
            f"Seg {seg_num + 1}\n({segment_length})",
            ha="left",
            va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=color, edgecolor="black", alpha=0.7
            ),
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig


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
