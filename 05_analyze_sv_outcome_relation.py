#!/usr/bin/env python
""" Visualise position-wise SHAP distributions stratified by prediction outcome
(TP/FP/TN/FN) and trace-length cohort (short/medium/long).

Two plot styles are provided:
  - Violin  : seaborn horizontal violins; rows = cohorts, cols = samples.
  - Ridgeline: stacked KDE strips; one figure per cohort, cols = samples.
"""

import os
import pickle
import numpy as np
import pandas as pd
import os.path as osp
from typing import Dict, List, Optional

from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from experiments.setup_experiment import (
    load_data_and_model,
    create_loader_from_dataframe,
)
from ppm.utils import extract_explicands_samples
from local_xai.utils.trace_segmentation.transition_based import build_transition_matrix


# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]
PREFIX_LEN = 15
EXPLICANDS_NUM = 10
MAX_POSITIONS = 8   # segment positions above this are bucketed into "{N}+"

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")
config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth")
seg_output_dir = osp.join(OUTPUT_ROOT, "segmentation", "bpi17")
sv_output_dir = osp.join(OUTPUT_ROOT, "shap_values", "bpi17")

SAMPLE_META = {
    "tp": dict(label="TP\n(cancelled, correct)",  color="#3D7AACB6"),
    "fp": dict(label="FP\n(accepted, wrong)",     color="#DF9C38B6"),
    "fn": dict(label="FN\n(cancelled, wrong)",    color="#EA5348B6"),
    "tn": dict(label="TN\n(accepted, correct)",   color="#3E8540B6"),
}

COHORT_ORDER = ["short", "medium", "long"]


# ── Data builder ───────────────────────────────────────────────────────────────

def _label_sort_key(label: str) -> tuple:
    """Numeric sort key for position labels like 'pos 3' or 'pos 8+'."""
    num = label.split()[1]
    return (int(num.rstrip("+")), num.endswith("+"))


def _make_position_label(pos_idx: int, cap: Optional[int]) -> str:
    if cap is None or pos_idx < cap:
        return f"pos {pos_idx + 1}"
    return f"pos {cap}+"


def build_sv_position_df(
    explicands_info: Dict[str, Dict],
    max_positions: int = MAX_POSITIONS,
    per_cohort_max_positions: Optional[Dict[str, Optional[int]]] = None,
) -> pd.DataFrame:
    """Flatten nested SHAP results into a tidy DataFrame.

    Parameters
    ----------
    explicands_info : dict
        Nested as  [cohort][sample] -> info_dict.
        info_dict must have a 'sv' key: list of dicts with 'segment_sv' (1-D array).
    max_positions : int
        Default cap: segment positions beyond this index are bucketed into
        ``f"pos {max_positions}+"``.
    per_cohort_max_positions : dict, optional
        Override cap per cohort.  ``None`` as a value means no bucketing for
        that cohort (all positions shown individually).
        Example: ``{"medium": None, "long": 6}``

    Returns
    -------
    pd.DataFrame with columns: cohort, sample, seg_position (Categorical), shap_value.
    """
    overrides = per_cohort_max_positions or {}
    rows = []
    for cohort, cohort_data in explicands_info.items():
        cap = overrides.get(cohort, max_positions)
        for sample, info in cohort_data.items():
            for case_sv in info.get("sv", []):
                shap_values = np.asarray(case_sv["segment_sv"]).ravel()
                for pos_idx, sv in enumerate(shap_values):
                    rows.append({
                        "cohort":       cohort,
                        "sample":       sample,
                        "seg_position": _make_position_label(pos_idx, cap),
                        "shap_value":   float(sv),
                    })

    df = pd.DataFrame(rows)
    # derive ordered categories from the data so every cohort's labels are covered
    all_labels = sorted(df["seg_position"].unique(), key=_label_sort_key)
    df["seg_position"] = pd.Categorical(
        df["seg_position"], categories=all_labels, ordered=True,
    )
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────

_SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_red_blue",
    ["#0188FE", "#D4CAD6", "#FE0755"]  # blue → purple → red
)#cm.get_cmap("RdBu_r")   # red = positive SV, blue = negative SV


def _cohort_sample_subset(df: pd.DataFrame, cohort: str, sample: str) -> pd.DataFrame:
    return df[(df["cohort"] == cohort) & (df["sample"] == sample)]


def _present_positions(df: pd.DataFrame, sub: pd.DataFrame) -> List[str]:
    """Return categories present in *sub*, in global order."""
    present = set(sub["seg_position"].dropna())
    return [p for p in df["seg_position"].cat.categories if p in present]


def _shap_color_palette(
    sub: pd.DataFrame,
    positions: List[str],
    norm: Optional[TwoSlopeNorm] = None,
) -> dict:
    """Map each segment position to a colour derived from its median SHAP value.

    If *norm* is provided it is reused (allows a shared scale across panels).
    Otherwise a local symmetric norm is built from the data in *sub*.
    """
    medians = {
        pos: sub[sub["seg_position"] == pos]["shap_value"].median()
        for pos in positions
    }
    if norm is None:
        vmax = max((abs(v) for v in medians.values()), default=1.0)
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    return {pos: _SHAP_CMAP(norm(med)) for pos, med in medians.items()}


def _cohort_shap_norm(cohort_df: pd.DataFrame) -> TwoSlopeNorm:
    """Build a symmetric TwoSlopeNorm from all SHAP values in *cohort_df*."""
    vmax = cohort_df["shap_value"].abs().max()
    vmax = max(float(vmax), 1e-6)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


# ── Violin plot ────────────────────────────────────────────────────────────────

def plot_sv_position_violin(
    df: pd.DataFrame,
    cohorts: List[str] = None,
    output_path: str = "",
) -> List[Figure]:
    """Horizontal violin plot: x = SHAP value, y = segment position.

    One figure per cohort; columns = samples (TP / FP / FN / TN).
    Each violin is coloured by the median SHAP value of that position using a
    diverging RdBu_r palette (red = positive, blue = negative).
    """
    cohorts = [c for c in COHORT_ORDER if c in (cohorts or df["cohort"].unique())]
    samples = [s for s in SAMPLE_NAMES if s in df["sample"].unique()]
    n_cols = len(samples)

    figs: List[Figure] = []

    for cohort in cohorts:
        cohort_df = df[df["cohort"] == cohort]
        # shared norm so the colorbar is consistent across all panels
        norm = _cohort_shap_norm(cohort_df)

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(4.5 * n_cols + 0.6, max(3.5, 0.55 * MAX_POSITIONS)),
            sharex=False,
            sharey=True,
            squeeze=False,
        )

        for c, sample in enumerate(samples):
            ax = axes[0][c]
            sub = _cohort_sample_subset(cohort_df, cohort, sample)
            positions = _present_positions(df, sub)

            if sub.empty or not positions:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_yticks([])
                continue

            palette = _shap_color_palette(sub, positions, norm=norm)
            sns.violinplot(
                data=sub,
                x="shap_value",
                y="seg_position",
                order=positions,
                palette=palette,
                orient="h",
                inner="box",
                cut=0,
                linewidth=0.8,
                scale="width",
                ax=ax,
            )
            ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.55)

            # annotate sample counts per position on the right margin
            x_right = ax.get_xlim()[1]
            for yi, pos in enumerate(positions):
                n = sub[sub["seg_position"] == pos]["shap_value"].count()
                ax.annotate(
                    f"n={n}",
                    xy=(x_right, yi),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center", ha="left",
                    fontsize=7, color="#555555",
                    annotation_clip=False,
                )

            ax.set_title(SAMPLE_META[sample]["label"].replace("\n", " "), fontsize=11)
            ax.set_ylabel("Segment position" if c == 0 else "", fontsize=10)
            ax.set_xlabel("SHAP value", fontsize=10)
            sns.despine(ax=ax)

        fig.suptitle(
            f"Position-wise SHAP value distributions — cohort: {cohort}",
            fontsize=13, y=1.01,
        )
        fig.tight_layout()
        # Reserve a strip on the right for the colorbar *after* tight_layout
        # so it does not steal space from or overlap the panel axes.
        fig.subplots_adjust(right=0.85)
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.68])

        # ── shared colorbar: violin colour = median SHAP value ───────
        sm = ScalarMappable(cmap=_SHAP_CMAP, norm=Normalize(vmin=norm.vmin, vmax=norm.vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Median SHAP value\n(violin fill colour)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        if output_path:
            base, ext = osp.splitext(output_path)
            path = f"{base}_{cohort}{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")

        figs.append(fig)

    return figs


# ── Ridgeline plot ─────────────────────────────────────────────────────────────

def plot_sv_position_ridgeline(
    df: pd.DataFrame,
    cohorts: List[str] = None,
    output_path: str = "",
    overlap: float = 0.55,
    bw_method: str = "scott",
) -> List[Figure]:
    """Ridgeline (joy) plot: stacked KDE strips, one figure per cohort.

    Layout: one figure per cohort; columns = samples (TP / FP / FN / TN).
    x-axis = SHAP value, y = stacked segment-position KDEs.

    Parameters
    ----------
    overlap   : vertical overlap fraction between adjacent ridges.
    bw_method : bandwidth estimator passed to ``scipy.stats.gaussian_kde``.
    """
    cohorts = [c for c in COHORT_ORDER if c in (cohorts or df["cohort"].unique())]
    samples = [s for s in SAMPLE_NAMES if s in df["sample"].unique()]

    figs: List[Figure] = []

    for cohort in cohorts:
        cohort_df = df[df["cohort"] == cohort]
        fig, axes = plt.subplots(
            1, len(samples),
            figsize=(4.5 * len(samples), 5.5),
        )
        if len(samples) == 1:
            axes = [axes]

        for ax, sample in zip(axes, samples):
            sub = _cohort_sample_subset(cohort_df, cohort, sample)
            positions = _present_positions(df, sub)

            if sub.empty or not positions:
                ax.set_visible(False)
                continue

            color = SAMPLE_META[sample]["color"]
            row_height = 1.0

            # global x range for consistent axis limits
            x_all = sub["shap_value"].values
            pad = (x_all.max() - x_all.min()) * 0.10 + 1e-6
            x_min, x_max = x_all.min() - pad, x_all.max() + pad
            x_grid = np.linspace(x_min, x_max, 500)

            for i, pos in enumerate(reversed(positions)):
                vals = sub[sub["seg_position"] == pos]["shap_value"].values
                y_offset = i * row_height * (1.0 - overlap)

                if len(vals) < 3:
                    # draw a flat baseline for positions with too few observations
                    ax.axhline(y_offset, color=color, linewidth=0.6, alpha=0.4)
                    ax.text(x_min, y_offset, pos,
                            ha="right", va="bottom", fontsize=8,
                            clip_on=False)
                    continue

                kde = gaussian_kde(vals, bw_method=bw_method)
                density = kde(x_grid)
                # scale so the tallest ridge fills one row_height
                density = density / density.max() * row_height * (1.0 + overlap * 0.8)

                ax.fill_between(
                    x_grid,
                    y_offset,
                    y_offset + density,
                    alpha=0.72,
                    color=color,
                    linewidth=0,
                )
                ax.plot(x_grid, y_offset + density,
                        color=color, linewidth=0.9, alpha=0.95)
                # white baseline between ridges
                ax.fill_between(x_grid, y_offset - 0.04, y_offset,
                                color="white", linewidth=0)

                # position label left of the axis
                n = len(vals)
                ax.text(
                    x_min - (x_max - x_min) * 0.03,
                    y_offset,
                    f"{pos}  (n={n})",
                    ha="right", va="bottom",
                    fontsize=8, color="#333333",
                    clip_on=False,
                )

            ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6,
                       zorder=10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(-row_height * 0.3, None)
            ax.set_yticks([])
            ax.set_xlabel("SHAP value", fontsize=10)
            ax.set_title(SAMPLE_META[sample]["label"].replace("\n", " "), fontsize=11)
            sns.despine(ax=ax, left=True)

        # ── figure-level legend: panel colour = prediction outcome ───
        legend_handles = [
            mpatches.Patch(facecolor=SAMPLE_META[s]["color"], edgecolor="grey",
                           linewidth=0.5, label=SAMPLE_META[s]["label"].replace("\n", " "))
            for s in samples
        ]
        fig.legend(
            handles=legend_handles,
            title="Prediction outcome\n(panel colour)",
            title_fontsize=8,
            fontsize=8,
            loc="lower center",
            ncol=len(samples),
            bbox_to_anchor=(0.5, -0.04),
            frameon=True,
        )

        fig.suptitle(
            f"Position-wise SHAP distributions — cohort: {cohort}",
            fontsize=13, y=1.01,
        )
        fig.tight_layout()

        if output_path:
            base, ext = osp.splitext(output_path)
            path = f"{base}_{cohort}{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")

        figs.append(fig)

    return figs


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data & model ────────────────────────────────────────
    print("Loading data and model …")
    config, train_loader, test_loader, model = load_data_and_model(
        config_path, checkpoint_path
    )

    # ── 2. Load saved SHAP explanations per cohort ──────────────────
    cohorts = ["short", "medium", "long"]
    explicands_info: Dict[str, Dict] = {}

    for cohort in cohorts:
        sv_cohort_dir = osp.join(sv_output_dir, f"transition_cohort_{cohort}")
        explicands_info[cohort] = {}
        for name in SAMPLE_NAMES:
            pkl_path = osp.join(sv_cohort_dir, f"{name}_segment_sv_results.pkl")
            with open(pkl_path, "rb") as f:
                explicands_info[cohort][name] = pickle.load(f)

    print("Loaded SHAP explanations and segment info.")

    # ── 3. Build flat DataFrame ──────────────────────────────────────
    sv_pos_df = build_sv_position_df(
        explicands_info,
        max_positions=MAX_POSITIONS,
        per_cohort_max_positions={"medium": None},  # show every position for medium
    )
    print(f"SV position DataFrame shape: {sv_pos_df.shape}")
    print(
        sv_pos_df
        .groupby(["cohort", "sample", "seg_position"])["shap_value"]
        .describe()
        .round(4)
    )

    vis_dir = osp.join(OUTPUT_ROOT, "figures", "bpi17", "sv_position")
    os.makedirs(vis_dir, exist_ok=True)

    # ── 4a. Violin plot (one figure per cohort) ──────────────────────
    plot_sv_position_violin(
        sv_pos_df,
        cohorts=cohorts,
        output_path=osp.join(vis_dir, "sv_position_violin.png"),
    )

    # ── 4b. Ridgeline plot (one figure per cohort) ───────────────────
    # plot_sv_position_ridgeline(
    #     sv_pos_df,
    #     cohorts=cohorts,
    #     output_path=osp.join(vis_dir, "sv_position_ridgeline.png"),
    # )


if __name__ == "__main__":
    main()
