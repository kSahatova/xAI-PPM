import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Tuple


def visualize_trace_vertical(
    trace: np.ndarray,
    seg_boundaries_colors: Dict[Tuple[int, int], Tuple],
    activity_lookup: Dict[int, str],
    activity_token_index: int = 0,
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
    activity_tokens = trace[:, activity_token_index].astype(int)
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

    plt.tight_layout()
    plt.show()
