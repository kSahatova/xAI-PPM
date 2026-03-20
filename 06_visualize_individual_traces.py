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
from local_xai.utils.trace_segmentation.evaluation import (
    compute_er_full_and_segments, compute_er_full, compute_er_segments
)


# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]
TOP_N_PATTERNS = 10

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")


# config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op_sepsis.txt")
config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
# config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op_bpi15.txt")
# checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\Sepsis_rnn_outcome_sepsis.pth")
checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth")
# checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\rnn_outcome_bpi15_1.pth")

SAMPLE_META = {
    "tp": dict(label="TP\n(cancelled, correct)", color="#3D7AACB6"),
    "fp": dict(label="FP\n(accepted, wrong)", color="#DF9C38B6"),
    "fn": dict(label="FN\n(cancelled, wrong)", color="#EA5348B6"),
    "tn": dict(label="TN\n(accepted, correct)", color="#3E8540B6"),
}

SEG_STRATEGIES = ["transition"]
# SEG_STRATEGIES = ["per_event", "distribution", "transition"]

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


# ── Individual-trace chart ───────────────────────────────────────────────


def plot_individual_traces(
    explicands_info: Dict,
    activity_lookup: dict,
    sample: str,
    n_cases: int = 10,
    pattern_type: str = "positional",
    truncate_start: int = 0,
    truncate_end: int = 0,
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
    truncate_start : int
        Number of events to drop from the beginning of each trace (default 0).
    truncate_end : int
        Number of events to drop from the end of each trace (default 0).
        Together these two parameters let you zoom into the middle of long
        traces so that activity labels in the cells can be displayed larger.
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

    trace_len = max(
        max(sv_dict["segment_ids"][-1][-1] + 1 - truncate_start - truncate_end, 0)
        for _, sv_dict, *_ in selected
    )

    # Scale cell label font with available width per cell
    fig_w = max(14, trace_len * 0.6 + 4)
    pts_per_cell = fig_w / max(trace_len, 1) * 72  # 72 pt/inch
    cell_font_size = min(max(int(pts_per_cell * 0.18), 9), 22)

    fig, ax = plt.subplots(
        figsize=(fig_w, n * (row_h + gap) + 2.0),
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
        vis_end = row_trace_len - truncate_end  # exclusive upper bound
        for event_idx in range(truncate_start, vis_end):
            display_x = (event_idx - truncate_start) * cell_w
            color = event_color.get(event_idx, (0.85, 0.85, 0.85, 1.0))
            rect = mpatches.Rectangle(
                (display_x, y0),
                cell_w,
                row_h,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            ax.add_patch(rect)
            act_code = int(case[0, event_idx, 0])
            act_name = activity_lookup.get(act_code, f"#{act_code}")
            r, g, b, _ = color
            txt_color = (
                "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"
            )
            txt = ax.text(
                display_x + cell_w / 2,
                y0 + row_h / 2,
                act_name,
                ha="center",
                va="center",
                fontsize=cell_font_size,
                rotation=90,
                color=txt_color,
                clip_on=True,
                zorder=3,
            )
            txt.set_clip_path(rect)  # type: ignore[arg-type]

        # Segment boundary lines — only those inside the visible window
        for cp in _change_points(seg_ids):
            if truncate_start < cp < vis_end:
                display_cp = (cp - truncate_start) * cell_w
                ax.plot(
                    [display_cp, display_cp],
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
    # x-ticks: display positions map back to original 1-indexed event numbers
    tick_display = np.arange(0, trace_len)
    left_margin = 0.0 if truncate_start > 0 else -5.0
    ax.set_xlim(left_margin, trace_len * cell_w)
    ax.set_ylim(-0.5, n * (row_h + gap) + 0.3)
    ax.set_xticks([i * cell_w + 0.5 for i in tick_display])
    ax.set_xlabel("Event position", fontsize=21, labelpad=6)
    ax.set_xticklabels(tick_display + truncate_start + 1, fontsize=19)
    # Centre the x-axis label over the cell data (0 … trace_len), not the full xlim
    xlim_l, xlim_r = left_margin, trace_len * cell_w
    label_x = (trace_len * cell_w / 2 - xlim_l) / (xlim_r - xlim_l)
    ax.xaxis.set_label_coords(label_x, -0.06)
    # ax.set_xticks([])
    ax.set_yticks([])
    # ax.tick_params(which="both", top=False, bottom=True, left=False, right=False)
    ax.grid(False)
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
    # ── 1. Load activity vocabulary once ────────────────────────────
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
    cohort = 'medium'


    for strategy in SEG_STRATEGIES:
        explicands_per_strategy[strategy] = {}
        print(f" [*] Loading '{strategy}' explanations — cohort '{cohort.upper()}'")
        sv_strategy_dir = osp.join(sv_output_dir, f'{strategy}_cohort_{cohort}')
        explicands_per_strategy[strategy][cohort] = {}
        for name in SAMPLE_NAMES:
            pkl_path = osp.join(sv_strategy_dir, f"{name}_segment_sv_results.pkl")
            with open(pkl_path, "rb") as f:
                explicands_per_strategy[strategy][cohort][name] = pickle.load(f)

    # ── 2. Evaluate entropic relevance ────────────────────────────
    print("\n" + "=" * 80)
    print("Entropic Relevance — full traces vs. segments")
    print("-" * 80)
    strategy = "transition"

    all_cases = []
    for _, item in explicands_per_strategy[strategy]['medium'].items():
        all_cases.extend(item['cases'])
    full_log_er, full_log_er_std = compute_er_full(all_cases, normalized=False)
    print(f"ER of the full log: {full_log_er} ({full_log_er_std})")


    for strategy, cohort_data in explicands_per_strategy.items():
        
        case_segments = []
        
        for _, sample_data in cohort_data.items():
            for sample in SAMPLE_NAMES:
                segments_info = sample_data[sample]['segments']
                segments = [item['segments'] for item in segments_info]
                case_segments.extend(segments)

        seg_er, seg_er_std = compute_er_segments(case_segments, normalized=False) 
        print(
            f"ER of the segmented log '{strategy}': {seg_er}, ({seg_er_std})", 
        )

    print("=" * 80 + "\n")

    # ── 3. Individual-trace strip charts ────────────────────────────
    vis_dir = osp.join(OUTPUT_ROOT, "figures", ds_name, "sv_patterns")
    # vis_dir = osp.join(OUTPUT_ROOT, "figures", ds_name, "sv_patterns", "seg_comparison")
    os.makedirs(vis_dir, exist_ok=True)

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
                n_cases=3,
                truncate_start=5,
                truncate_end=5,
                pattern_type=pattern_type,
                output_path=osp.join(
                    vis_dir,
                    f"{cohort}_{strategy}_top3_patterns_{sample}.png",
                ),
            )
        print_segment_stats_latex(
            explicands_per_strategy[strategy][cohort],
            top_k=10,
            output_path=osp.join(
                sv_output_dir,
                f"{strategy}_cohort_{cohort}",
                f"{ds_name}_{strategy}_seg_stats.tex",
            ),
        )


if __name__ == "__main__":
    main()
