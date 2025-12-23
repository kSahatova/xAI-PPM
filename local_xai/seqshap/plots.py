import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_segment_shap_heatmap(segment_names, segment_level_sv, save_path=None):
    """
    Plot heatmap of segment-level SHAP values.

    Args:
        segment_names: List of segment names
        segment_level_sv: Array of SHAP values for each segment
    """

    colors = ["#5f8fd6", "#99c3fb", "#f5f5f5", "#ffaa92", "#d16f5b"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap_shap_cmap", colors)

    fig, ax = plt.subplots(figsize=(2, 4))
    sns.heatmap(
        segment_level_sv.reshape(-1, 1),
        annot=True,
        fmt=".4f",
        cmap=custom_cmap,
        center=0,
        yticklabels=segment_names,
        ax=ax,
        linewidth=0.5,
        linecolor="white",
    )
    ax.set_xlabel("SHAP values")
    # ax.set_title("Segment-level SHAP value explanations")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig
