#!/usr/bin/env python
"""Tile charts of average segment-level SHAP values for the most frequent
change-point patterns, stratified by prediction outcome (TP/FP/FN/TN) and
trace-length cohort (short / medium / long).

For each (cohort × sample) group the top-N most frequent change-point patterns
are identified.  All traces that share the same pattern are averaged per
segment position, and the result is shown as a heatmap tile chart.

  rows    = most frequent change-point patterns (sorted by descending count)
  columns = segment positions (seg 1, seg 2, …)
  colour  = mean SHAP value across traces that share the pattern
  cells   = annotated with the mean SHAP value
"""

import os
import pickle
import numpy as np
import pandas as pd
import os.path as osp
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from experiments.setup_experiment import load_data_and_model


# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]
TOP_N_PATTERNS = 10

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")


config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op_sepsis.txt")
# config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\Sepsis_rnn_outcome_sepsis.pth")
# checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth")

SAMPLE_META = {
    "tp": dict(label="TP\n(cancelled, correct)", color="#3D7AACB6"),
    "fp": dict(label="FP\n(accepted, wrong)", color="#DF9C38B6"),
    "fn": dict(label="FN\n(cancelled, wrong)", color="#EA5348B6"),
    "tn": dict(label="TN\n(accepted, correct)", color="#3E8540B6"),
}

SEG_STRATEGIES = ["per_event", "distribution", "transition"]

COHORT_ORDER = ["medium"]
# COHORT_ORDER = ["short", "medium", "long"]

_SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_red_blue",
    ["#0188FE", "#D4CAD6", "#FE0755"],  # blue → neutral → red
)


# ── Data builders ──────────────────────────────────────────────────────────────


def _change_points(segment_ids: list) -> Tuple[int, ...]:
    """Start index of every segment after the first (= the breakpoints)."""
    return tuple(seg[0] for seg in segment_ids[1:])


def _activity_change_points(
    sv_dict: dict,
    case: np.ndarray,
    feature_idx: int = 0,
) -> Tuple[Tuple[int, int], ...]:
    """Pattern as a tuple of (from, to) activity-code pairs at each segment boundary.

    For boundary between segment i and segment i+1:
        from = activity code of the *last* event in segment i
        to   = activity code of the *first* event in segment i+1

    Unlike ``_change_points``, which encodes *where* a transition occurs,
    this captures *what activity pair* marks each boundary.  Two traces
    whose change-points fall at different positions are grouped together
    whenever their boundary transition pairs match in the same order.
    """
    seg_ids = sv_dict["segment_ids"]
    return tuple(
        (
            int(case[0, seg_ids[i][-1], feature_idx]),
            int(case[0, seg_ids[i + 1][0], feature_idx]),
        )
        for i in range(len(seg_ids) - 1)
    )


def build_pattern_sv_df(explicands_info: Dict[str, Dict]) -> pd.DataFrame:
    """Build a long-format DataFrame with one row per (trace × segment).

    Columns
    -------
    cohort, sample, trace_id, pattern, seg_idx, shap_value
        trace_id  – unique integer within each (cohort, sample) group
        pattern   – tuple of change-point positions, e.g. (5, 9, 14)
        seg_idx   – 0-based segment position within the trace
        shap_value – segment-level SHAP value
    """
    rows = []
    for cohort, cohort_data in explicands_info.items():
        for sample, info in cohort_data.items():
            for trace_id, sv_dict in enumerate(info.get("sv", [])):
                seg_ids = sv_dict["segment_ids"]
                shap_values = np.asarray(sv_dict["segment_sv"]).ravel()
                cp = _change_points(seg_ids)
                for seg_idx, sv in enumerate(shap_values):
                    rows.append(
                        {
                            "cohort": cohort,
                            "sample": sample,
                            "trace_id": trace_id,
                            "pattern": cp,
                            "seg_idx": seg_idx,
                            "shap_value": float(sv),
                        }
                    )

    return pd.DataFrame(rows)


def _top_n_patterns(
    df: pd.DataFrame,
    cohort: str,
    sample: str,
    top_n: int,
) -> List[Tuple]:
    """Return the *top_n* patterns sorted by descending trace count."""
    sub = df[(df["cohort"] == cohort) & (df["sample"] == sample)]
    counts = (
        sub.drop_duplicates(subset=["trace_id", "pattern"])
        .groupby("pattern")
        .size()
        .sort_values(ascending=False)
    )
    return counts.head(top_n).index.tolist()


def build_heatmap_matrix(
    df: pd.DataFrame,
    cohort: str,
    sample: str,
    top_n: int = TOP_N_PATTERNS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build a (patterns × segment_positions) matrix of mean SHAP values.

    Returns
    -------
    matrix : DataFrame  – rows = pattern labels, cols = "seg 1", "seg 2", …
                          NaN where a segment position does not exist for the pattern.
    counts : Series     – trace count per pattern (same row order as *matrix*).
    """
    sub = df[(df["cohort"] == cohort) & (df["sample"] == sample)]
    patterns = _top_n_patterns(df, cohort, sample, top_n)

    if not patterns:
        return pd.DataFrame(), pd.Series(dtype=int)

    # trace counts per pattern
    trace_counts = (
        sub.drop_duplicates(subset=["trace_id", "pattern"]).groupby("pattern").size()
    )

    # mean SHAP per (pattern, seg_idx) — only for the top patterns
    pat_sub = sub[sub["pattern"].isin(patterns)]
    pivot = (
        pat_sub.groupby(["pattern", "seg_idx"])["shap_value"].mean().unstack("seg_idx")
    )
    pivot.columns = [f"seg {int(c) + 1}" for c in pivot.columns]

    # reorder rows by frequency and attach readable labels
    pivot = pivot.reindex(patterns)

    def _row_label(pat: tuple) -> str:
        cp_str = ", ".join(str(c) for c in pat) if pat else "—"
        n = trace_counts.get(pat, 0)
        return f"[{cp_str}]  (n={n})"

    labels = [_row_label(p) for p in patterns]
    pivot.index = labels
    counts = pd.Series({_row_label(p): trace_counts.get(p, 0) for p in patterns})

    return pivot, counts


# ── Tile chart ─────────────────────────────────────────────────────────────────


def plot_tile_charts(
    df: pd.DataFrame,
    cohorts: Optional[List[str]] = None,
    top_n: int = TOP_N_PATTERNS,
    output_path: str = "",
) -> List[Figure]:
    """Tile chart: mean SHAP value per segment for the top-N most frequent
    change-point patterns.

    One figure per cohort; columns = samples (TP / FP / FN / TN).
    """
    cohorts = [c for c in COHORT_ORDER if c in (cohorts or df["cohort"].unique())]
    samples = [s for s in SAMPLE_NAMES if s in df["sample"].unique()]

    # symmetric colour scale shared across the entire dataset
    vmax = max(float(df["shap_value"].abs().max()), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    figs: List[Figure] = []

    for cohort in cohorts:
        fig, axes = plt.subplots(
            1,
            len(samples),
            figsize=(5.5 * len(samples) + 1.2, 0.5 * top_n + 2.5),
            squeeze=False,
        )

        for c, sample in enumerate(samples):
            ax = axes[0][c]
            matrix, _ = build_heatmap_matrix(df, cohort, sample, top_n)

            if matrix.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="gray",
                )
                ax.set_axis_off()
                continue

            sns.heatmap(
                matrix,
                ax=ax,
                cmap=_SHAP_CMAP,
                norm=norm,
                annot=True,
                fmt=".3f",
                annot_kws={"size": 7.5},
                linewidths=0.4,
                linecolor="#cccccc",
                cbar=False,
                mask=matrix.isna(),
            )

            ax.set_title(
                SAMPLE_META[sample]["label"].replace("\n", " "),
                fontsize=11,
            )
            ax.set_xlabel("Segment position", fontsize=10)
            ax.set_ylabel("Change-point pattern" if c == 0 else "", fontsize=10)
            ax.tick_params(axis="x", rotation=0, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=7.5)

        # ── shared colorbar ────────────────────────────────────────────
        sm = ScalarMappable(
            cmap=_SHAP_CMAP,
            norm=Normalize(vmin=-vmax, vmax=vmax),
        )
        sm.set_array([])
        fig.tight_layout()
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.68])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Mean SHAP value", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(
            f"Most frequent segment patterns — cohort: {cohort}  (top {top_n})",
            fontsize=13,
            y=1.01,
        )

        if output_path:
            base, ext = osp.splitext(output_path)
            path = f"{base}_{cohort}{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")

        figs.append(fig)

    return figs


# ── Activity-segment tile chart ────────────────────────────────────────────────


def _build_sample_segment_summary(
    sv_list: list,
    cases_list: list,
    activity_lookup: dict,
) -> Tuple[tuple, int, pd.DataFrame]:
    """For the most common change-point pattern in *sample*, compute the most
    frequent activity and its proportion for every segment position.

    Returns
    -------
    pattern  : tuple of change-point indices (the most common pattern)
    n_traces : number of traces that share this pattern
    summary  : DataFrame indexed by seg_idx with columns
               ['top_activity', 'proportion']
    """
    patterns = [_change_points(sv["segment_ids"]) for sv in sv_list]
    pattern_counts = Counter(patterns)
    pattern = pattern_counts.most_common(1)[0][0]
    n_traces = pattern_counts[pattern]

    # Collect every (segment_position → activity name) occurrence
    seg_activities: Dict[int, List[str]] = defaultdict(list)
    for trace_id, sv_dict in enumerate(sv_list):
        if _change_points(sv_dict["segment_ids"]) != pattern:
            continue
        case = cases_list[trace_id]  # shape (1, prefix_len, n_features)
        for seg_idx, event_indices in enumerate(sv_dict["segment_ids"]):
            for event_idx in event_indices:
                act_code = int(case[0, event_idx, 0])
                act_name = activity_lookup.get(act_code, f"#{act_code}")
                seg_activities[seg_idx].append(act_name)

    rows = []
    for seg_idx in sorted(seg_activities):
        acts = seg_activities[seg_idx]
        top_act, top_count = Counter(acts).most_common(1)[0]
        rows.append(
            {
                "seg_idx": seg_idx,
                "top_activity": top_act,
                "proportion": top_count / len(acts),
            }
        )

    return pattern, n_traces, pd.DataFrame(rows).set_index("seg_idx")


def plot_activity_segment_tile(
    explicands_info: Dict,
    activity_lookup: dict,
    cohort: str = "short",
    output_path: str = "",
) -> Figure:
    """Tile chart: rows = prediction outcomes (TP/FP/FN/TN),
    columns = segment positions.

    Each cell shows the most frequent activity in that segment (for traces
    whose segmentation matches the most common change-point pattern of that
    outcome group).  Cell colour encodes the proportion of events in that
    segment that are the top activity; the activity name is printed inside
    the cell.

    Parameters
    ----------
    explicands_info : dict
        Loaded from pickle; keyed by sample name (tp/fp/fn/tn).
    activity_lookup : dict
        Maps encoded integer → activity name string
        (``test_loader.dataset.log.itos["activity"]``).
    cohort : str
        Label used only in the figure title.
    output_path : str
        If non-empty the figure is saved to this path.
    """
    samples = [s for s in SAMPLE_NAMES if s in explicands_info]

    # ── 1. Per-sample summaries ─────────────────────────────────────
    sample_data: Dict[str, Tuple[tuple, int, pd.DataFrame]] = {}
    max_seg_idx = 0
    for sample in samples:
        pattern, n_traces, summary = _build_sample_segment_summary(
            explicands_info[sample].get("sv", []),
            explicands_info[sample].get("cases", []),
            activity_lookup,
        )
        sample_data[sample] = (pattern, n_traces, summary)
        if not summary.empty:
            max_seg_idx = max(max_seg_idx, int(summary.index.max()))

    n_segs = max_seg_idx + 1
    col_labels = [f"seg {i + 1}" for i in range(n_segs)]

    # ── 2. Build proportion & annotation matrices ───────────────────
    prop_matrix = pd.DataFrame(np.nan, index=samples, columns=col_labels)
    annot_matrix = pd.DataFrame("", index=samples, columns=col_labels)

    for sample in samples:
        pattern, n_traces, summary = sample_data[sample]
        props = {int(k): float(v) for k, v in summary["proportion"].items()}  # type: ignore[arg-type]
        acts = {int(k): str(v) for k, v in summary["top_activity"].items()}  # type: ignore[arg-type]
        for seg_idx, proportion in props.items():
            col = f"seg {seg_idx + 1}"
            act = acts[seg_idx]
            act_short = act[:20] + "…" if len(act) > 20 else act
            prop_matrix.loc[sample, col] = proportion
            annot_matrix.loc[sample, col] = f"{act_short}\n{proportion:.0%}"

    # ── 3. Build readable row labels ────────────────────────────────
    row_labels = []
    for sample in samples:
        pattern, n_traces, _ = sample_data[sample]
        cp_str = ", ".join(str(c) for c in pattern) if pattern else "—"
        meta = SAMPLE_META[sample]["label"].replace("\n", " ")
        row_labels.append(f"{meta}\n[{cp_str}]  n={n_traces}")
    prop_matrix.index = row_labels
    annot_matrix.index = row_labels

    # ── 4. Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, 2.8 * n_segs + 2), 1.8 * len(samples) + 1.5))

    sns.heatmap(
        prop_matrix.astype(float),
        ax=ax,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annot=annot_matrix.values,
        fmt="",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="#cccccc",
        cbar_kws={"label": "Proportion of top activity\n(among events in segment)"},
        mask=prop_matrix.isna(),
    )

    ax.set_title(
        f"Most frequent activity per segment — cohort: {cohort}",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Segment position", fontsize=10)
    ax.set_ylabel("Prediction outcome  [pattern]", fontsize=10)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")

    return fig


# ── Individual-trace strip chart ───────────────────────────────────────────────


def plot_individual_traces(
    explicands_info: Dict,
    activity_lookup: dict,
    sample: str,
    cohort: str = "short",
    n_cases: int = 10,
    pattern_type: str = "positional",
    output_path: str = "",
) -> Figure:
    """Draw individual traces as per-event cell strips (one cell = one event).

    All cells that belong to the same segment share the same SHAP-derived fill
    colour.  Segment boundaries are drawn as thick vertical lines.
    A shared diverging colour-bar sits flush on the right.

    Parameters
    ----------
    pattern_type : {"positional", "activity"}
        How traces are grouped into patterns:
        - "positional" : by the tuple of change-point *positions* (default)
        - "activity"   : by the tuple of (from_act, to_act) pairs at each
                         segment boundary — position-invariant grouping
    """
    sv_list = explicands_info[sample].get("sv", [])
    cases_list = explicands_info[sample].get("cases", [])

    # ── 1. Compute patterns and select one representative per top-k ───
    if pattern_type == "activity":
        all_patterns = [
            _activity_change_points(sv, cases_list[i]) for i, sv in enumerate(sv_list)
        ]
    else:
        all_patterns = [_change_points(sv["segment_ids"]) for sv in sv_list]

    pattern_counts = Counter(all_patterns)
    top_patterns = [pat for pat, _ in pattern_counts.most_common(n_cases)]

    # total_traces = len(all_patterns)
    # covered      = sum(pattern_counts[p] for p in top_patterns)
    # print(f"[{pattern_type}] Top-{len(top_patterns)} patterns cover "
    #       f"{covered}/{total_traces} traces ({100 * covered / total_traces:.1f}%)")

    # For each top pattern pick the first matching trace
    selected: List[
        Tuple[int, dict, tuple, int]
    ] = []  # (tid, sv_dict, pattern, pat_count)
    seen: set = set()
    for tid in range(len(sv_list)):
        pat = all_patterns[tid]
        if pat in top_patterns and pat not in seen:
            seen.add(pat)
            selected.append((tid, sv_list[tid], pat, pattern_counts[pat]))
        if len(selected) == len(top_patterns):
            break
    # Sort by descending pattern frequency so the most common is on top
    selected.sort(key=lambda x: -x[3])

    if not selected:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "no data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )
        return fig

    # ── 2. Shared SHAP colour scale ──────────────────────────────────
    all_svs = [
        float(v)
        for _, sv_dict, *_ in selected
        for v in np.asarray(sv_dict["segment_sv"]).ravel()
    ]
    vmax = max(abs(min(all_svs)), abs(max(all_svs)), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # ── 3. Layout constants ──────────────────────────────────────────
    cell_w = 1.0  # width of one event cell (data units)
    row_h = 1.4  # height of each trace row
    gap = 0.3  # vertical gap between rows
    n = len(selected)

    trace_len = max(sv_dict["segment_ids"][-1][-1] + 1 for _, sv_dict, *_ in selected)

    fig, ax = plt.subplots(
        figsize=(max(14, trace_len * 0.6 + 4), n * (row_h + gap) + 2.0),
        facecolor="white",
    )
    ax.set_facecolor("white")

    # ── 4. Draw rows ─────────────────────────────────────────────────
    for row_idx, (trace_id, sv_dict, row_pattern, row_count) in enumerate(selected):
        y0 = (n - 1 - row_idx) * (row_h + gap)
        case = cases_list[trace_id]  # shape (1, prefix_len, n_features)
        seg_ids = sv_dict["segment_ids"]
        shap_vals = np.asarray(sv_dict["segment_sv"]).ravel()

        # Build event → segment colour lookup
        event_color: Dict[int, tuple] = {}
        for seg_idx, (event_indices, sv) in enumerate(zip(seg_ids, shap_vals)):
            color = _SHAP_CMAP(norm(float(sv)))
            for event_idx in event_indices:
                event_color[event_idx] = color

        # One coloured cell per event; show integer activity code
        row_trace_len = seg_ids[-1][-1] + 1
        for event_idx in range(row_trace_len):
            color = event_color.get(event_idx, (0.85, 0.85, 0.85, 1.0))
            ax.add_patch(
                mpatches.Rectangle(
                    (event_idx * cell_w, y0),
                    cell_w,
                    row_h,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0,
                    zorder=2,
                )
            )
            act_code = int(case[0, event_idx, 0])
            act_name = activity_lookup.get(act_code, f"#{act_code}")
            r, g, b, _ = color
            txt_color = (
                "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"
            )
            ax.text(
                event_idx * cell_w + cell_w / 2,
                y0 + row_h / 2,
                act_name,
                ha="center",
                va="center",
                fontsize=9,
                rotation=90,
                color=txt_color,
                clip_on=True,
                zorder=3,
            )

        # Segment boundary lines — always positional regardless of pattern_type
        for cp in _change_points(seg_ids):
            ax.plot(
                [cp * cell_w, cp * cell_w],
                [y0, y0 + row_h],
                color="#222222",
                linewidth=2.0,
                zorder=4,
            )

        # Row label
        if pattern_type == "activity":
            # transitions = "  ".join(f"{a}→{b}" for a, b in row_pattern)
            label = f"Pattern {row_idx + 1}\n$(n={row_count})$"
        else:
            label = f"Pattern {row_idx + 1}\n$(n={row_count})$"
        ax.text(
            -0.5,
            y0 + row_h / 2,
            label,
            ha="right",
            va="center",
            fontsize=19,
            linespacing=1.4,
        )

    # ── 5. Axes ───────────────────────────────────────────────────────
    # x-ticks every 5 events, 1-indexed, at left edge of each labelled cell
    tick_positions = np.arange(0, trace_len)
    # ax.set_xlim(-5.0, trace_len * cell_w)
    ax.set_ylim(-0.5, n * (row_h + gap) + 0.3)
    ax.set_xticks([i * cell_w + 0.5 for i in tick_positions])
    ax.set_xticklabels(tick_positions + 1, fontsize=19)
    ax.set_xlabel("Event position", fontsize=21, labelpad=6)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── 6. Colorbar as inset — stays flush with the plot ─────────────
    sm = ScalarMappable(cmap=_SHAP_CMAP, norm=Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cax = ax.inset_axes([1.01, 0.05, 0.02, 0.9])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Segment SHAP value", fontsize=19)
    cbar.ax.tick_params(labelsize=19)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")

    return fig


# ── Segment statistics LaTeX table ────────────────────────────────────────────


def print_segment_stats_latex(
    explicands_info: Dict, top_k: int = 5, output_path: str = ""
) -> None:
    """Save a LaTeX table with segment statistics for each sample (TP/FP/FN/TN).

    Rows    : TP, FP, FN, TN
    Columns : sample size (N), # segments (mean ± std),
              segment length (mean ± std),
              top-k positional-pattern coverage (%),
              top-k activity-transition-pattern coverage (%)
    """
    table_rows = []
    for sample in SAMPLE_NAMES:
        if sample not in explicands_info:
            continue
        sv_list = explicands_info[sample].get("sv", [])
        if not sv_list:
            continue

        n_segs_per_trace: List[float] = []
        seg_len_per_trace: List[float] = []
        for sv_dict in sv_list:
            seg_ids = sv_dict["segment_ids"]
            n_segs_per_trace.append(len(seg_ids))
            seg_len_per_trace.append(float(np.mean([len(seg) for seg in seg_ids])))

        cases_list = explicands_info[sample].get("cases", [])

        # positional coverage: pattern = change-point indices
        patterns = [_change_points(sv["segment_ids"]) for sv in sv_list]
        pattern_counts = Counter(patterns)
        top_patterns = [pat for pat, _ in pattern_counts.most_common(top_k)]
        total_traces = len(patterns)
        covered_traces = sum(pattern_counts[pat] for pat in top_patterns)
        coverage_pct = (
            100.0 * covered_traces / total_traces if total_traces > 0 else 0.0
        )

        # activity-transition coverage: pattern = activity codes at segment boundaries
        act_patterns = [
            _activity_change_points(sv, cases_list[i]) for i, sv in enumerate(sv_list)
        ]
        act_pattern_counts = Counter(act_patterns)
        act_top = [pat for pat, _ in act_pattern_counts.most_common(top_k)]
        act_covered = sum(act_pattern_counts[p] for p in act_top)
        act_coverage_pct = (
            100.0 * act_covered / total_traces if total_traces > 0 else 0.0
        )

        table_rows.append(
            (
                sample.upper(),
                total_traces,
                float(np.mean(n_segs_per_trace)),
                float(np.std(n_segs_per_trace)),
                float(np.mean(seg_len_per_trace)),
                float(np.std(seg_len_per_trace)),
                coverage_pct,
                act_coverage_pct,
                len(act_pattern_counts),
            )
        )

    lines = [
        f"% Segment statistics per sample  (top-{top_k} patterns)",
        r"\begin{table}[h]",
        r"\centering",
        (
            f"\\caption{{Segment statistics per prediction outcome "
            f"(top-{top_k} patterns, $\\mu \\pm \\sigma$ across traces)}}"
        ),
        r"\begin{tabular}{lccccc}",
        r"\hline",
        (
            r"Sample & $N$ & \# segments ($\mu \pm \sigma$) & "
            r"Seg.\ length ($\mu \pm \sigma$) & "
            r"Top-$k$ cov.\ pos.\ (\%) & Top-$k$ cov.\ act.\ (\%) & "
            r"\# patterns\\"
        ),
        r"\hline",
    ]
    for sample_name, n, ns_m, ns_s, sl_m, sl_s, cov, act_cov, n_act_pat in table_rows:
        lines.append(
            f"{sample_name} & {n} & "
            f"${ns_m:.2f} \\pm {ns_s:.2f}$ & "
            f"${sl_m:.2f} \\pm {sl_s:.2f}$ & "
            f"${cov:.1f}\\%$ & "
            f"${act_cov:.1f}\\%$ &"
            f"${n_act_pat} $ \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    # ── 2. Load activity vocabulary once ────────────────────────────
    print("=" * 80)
    print("Loading model & vocabulary for activity decoding")
    print("=" * 80)
    config, _, test_loader, _ = load_data_and_model(config_path, checkpoint_path)
    ds_name = config["dataset"].lower()
    raw_stoi: dict = test_loader.dataset.log.stoi["activity"]  # type: ignore[union-attr]
    activity_lookup: dict = {v: k for k, v in raw_stoi.items()}  # int → activity name

    # ── 1. Load saved SHAP explanations per strategy × cohort ───────
    explicands_per_strategy: Dict[str, Dict[str, Dict]] = {}

    sv_output_dir = osp.join(OUTPUT_ROOT, "shap_values", ds_name)

    for strategy in SEG_STRATEGIES:
        explicands_per_strategy[strategy] = {}
        for cohort in COHORT_ORDER:
            print(f"[*] Loading '{strategy}' explanations — cohort '{cohort.upper()}'")
            sv_cohort_dir = osp.join(sv_output_dir, f"{strategy}_cohort_{cohort}")
            explicands_per_strategy[strategy][cohort] = {}
            for name in SAMPLE_NAMES:
                pkl_path = osp.join(sv_cohort_dir, f"{name}_segment_sv_results.pkl")
                with open(pkl_path, "rb") as f:
                    explicands_per_strategy[strategy][cohort][name] = pickle.load(f)

    # ── 3. Individual-trace strip charts ────────────────────────────
    vis_dir = osp.join(OUTPUT_ROOT, "figures", ds_name, "sv_patterns", "seg_comparison")
    os.makedirs(vis_dir, exist_ok=True)

    cohort = COHORT_ORDER[0]
    pattern_type = "activity"

    for strategy in SEG_STRATEGIES:
        print(f"{'=' * 80}")
        print(f"Visualizing strategy: {strategy}")
        print(f"{'=' * 80}")
        for sample in SAMPLE_NAMES:
            plot_individual_traces(
                explicands_per_strategy[strategy][cohort],
                activity_lookup,
                sample=sample,
                cohort=cohort,
                n_cases=1,
                pattern_type=pattern_type,
                output_path=osp.join(
                    vis_dir,
                    f"{cohort}_{strategy}_individual_traces_{sample}_{pattern_type}.png",
                ),
            )
        print_segment_stats_latex(
            explicands_per_strategy[strategy][cohort],
            top_k=10,
            output_path=osp.join(
                sv_output_dir,
                f"{strategy}_cohort_{cohort}",
                f"{ds_name}_{}_seg_stats.tex",
            ),
        )


if __name__ == "__main__":
    main()
