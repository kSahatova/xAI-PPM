import os.path as osp
import shap
from shap import Explanation

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_segment_shap_heatmap(
    segment_names,
    segment_level_sv,
    plot_hor: bool = False,
    figsize=(4, 2),
    save_path=None,
):
    """
    Plot heatmap of segment-level SHAP values.

    Args:
        segment_names: List of segment names
        segment_level_sv: Array of SHAP values for each segment
    """

    colors = ["#5f8fd6", "#99c3fb", "#f5f5f5", "#ffaa92", "#d16f5b"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_heatmap_shap_cmap", colors)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        segment_level_sv,
        annot=True,
        fmt=".4f",
        cmap=custom_cmap,
        center=0,
        ax=ax,
        linewidth=0.5,
        linecolor="white",
    )

    if plot_hor:
        # ax.tick_params(left=False)
        ax.set_xticklabels(segment_names)
        ax.set_yticklabels(labels=[])
    else:
        ax.set_xlabel("SHAP values")
        ax.set_yticklabels(segment_names)

    # ax.set_title("Segment-level SHAP value explanations")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_segment_level_sv(seg_sv, seg_explainer, seg_names, i, output_dir):
    expl = Explanation(
        seg_sv,
        base_values=seg_explainer.expected_value,
        feature_names=seg_names,
    )
    fig = shap.plots.force(expl, matplotlib=True, contribution_threshold=0, show=False)
    fig.savefig(
        osp.join(output_dir, f"case_{i}_seg_sv_force.png"),
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    hm = plot_segment_shap_heatmap(seg_names, seg_sv)
    hm.savefig(
        osp.join(output_dir, f"case_{i}_seg_sv_heatmap.png"),
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close("all")


def plot_feature_level_sv(feat_sv, feat_explainer, config, i, output_dir):
    expl = Explanation(
        feat_sv,
        base_values=feat_explainer.expected_value,
        feature_names=config["categorical_features"] + config["continuous_features"],
    )
    plt.figure()
    shap.plots.bar(expl, show=False)
    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, f"case_{i}_feat_sv_bar.png"),
        dpi=300,
        facecolor="white",
    )
    plt.close("all")
