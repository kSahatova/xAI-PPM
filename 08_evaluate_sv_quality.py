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
from typing import Dict, List, Tuple

from experiments.setup_experiment import load_data_and_model
from local_xai.utils.baseline_calculation import build_average_event_baseline
from timeshap.wrappers.outcome_predictor_wrapper import OutcomePredictorWrapper


# ── Constants ──────────────────────────────────────────────────────────────────
SEG_STRATEGIES = ["per_event", "distribution", "transition"]
RANDOM_SEEDS = [0]
# RANDOM_SEEDS = [6, 32, 42, 105]
COHORT_ORDER = ["medium"]
SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]
TOP_N_PATTERNS = 10
AVERAGE_CP_NUM = 8
# When True, per-event monotonicity is re-grouped by transition-based change
# points before evaluation ("grouped_mono").  When False, each event is its
# own segment ("mono").
USE_GROUPED_MONO_FOR_PER_EVENT = False

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")
sv_output_dir = osp.join(OUTPUT_ROOT, "shap_values", "bpi17")

config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth")

SAMPLE_META = {
    "tp": dict(label="TP\n(cancelled, correct)", color="#3D7AACB6"),
    "fp": dict(label="FP\n(accepted, wrong)", color="#DF9C38B6"),
    "fn": dict(label="FN\n(cancelled, wrong)", color="#EA5348B6"),
    "tn": dict(label="TN\n(accepted, correct)", color="#3E8540B6"),
}


_ROW_ORDER = ["tp", "fp", "fn", "tn", "predicted_positive", "predicted_negative"]
_ROW_LABELS = {
    "tp": "TP (cancelled, correct)",
    "fp": "FP (accepted, wrong)",
    "fn": "FN (cancelled, wrong)",
    "tn": "TN (accepted, correct)",
    "predicted_positive": "Predicted positive",
    "predicted_negative": "Predicted negative",
}
_STRATEGY_LABELS = {
    "per_event": "Per-event segmentation",
    "random": "Random segmentation",
    "distribution": "Distribution-based segmentation",
    "transition": "Transition-based segmentation",
}


# ── RPCi / RPCu ─────────────────────────────────────────────────────────────────

def _perturb_segments(
    case: np.ndarray,
    segment_ids: list,
    perturb_seg_indices: list,
    avg_event: np.ndarray,
) -> np.ndarray:
    """Return a copy of *case* with the specified segments replaced by *avg_event*.

    Parameters
    ----------
    case : (1, L, D) ndarray
    segment_ids : list of lists – timestep indices per segment,
                  e.g. [[0, 1, 2], [3, 4, 5], ...]
    perturb_seg_indices : positions in segment_ids to perturb
    avg_event : (1, D) ndarray – average-event baseline
    """
    x = case.copy()
    for seg_idx in perturb_seg_indices:
        for t in segment_ids[seg_idx]:
            x[0, t, :] = avg_event[0, :]
    return x


def compute_monotonicity_for_case(
    case: np.ndarray,
    y_hat: float,
    segment_sv: np.ndarray,
    segment_ids: list,
    avg_event: np.ndarray,
    fetching_fn,
) -> float:
    """Compute the monotonicity score for a single instance.

    Starts from the original trace and perturbs segments one by one in
    descending order of |SHAP| importance.  At each step the distance to the
    original prediction Δ_i = |f(σ) − f(σ^(i))| should increase — the
    fraction of consecutive pairs (Δ_i, Δ_{i+1}) where Δ_{i+1} > Δ_i is
    the monotonicity score.

    Parameters
    ----------
    case : (1, L, D) ndarray
    y_hat : scalar – original model prediction probability
    segment_sv : (n_segments,) ndarray – segment SHAP values
    segment_ids : list of lists – timestep indices per segment
    avg_event : (1, D) ndarray – perturbation baseline
    fetching_fn : wrapped model; f(x) -> (predictions (B,1), hidden_state)

    Returns
    -------
    float in [0, 1] – fraction of monotonically increasing distance steps,
    or np.nan when fewer than 2 segments exist (score is undefined).
    """
    n_segments = len(segment_ids)
    if n_segments < 2:
        return np.nan

    # Rank segments by descending |SHAP|
    order = np.argsort(np.abs(segment_sv.flatten()))[::-1].tolist()

    # Perturb one by one, computing Δ_i = |y_hat - f(σ^(i))| after each removal
    delta = [0.0]
    current = case.copy()
    for seg_idx in order:
        for t in segment_ids[seg_idx]:
            current[0, t, :] = avg_event[0, :]
        pred, _ = fetching_fn(current)
        delta.append(abs(y_hat - float(pred[0, 0])))

    # Count consecutive pairs where distance strictly increased
    monotonic_steps = sum(
        1 for i in range(1, n_segments) if delta[i + 1] > delta[i]
    )
    return monotonic_steps / (n_segments - 1)


def compute_monotonicity_grouped_for_case(
    case: np.ndarray,
    y_hat: float,
    per_event_sv: np.ndarray,
    per_event_seg_ids: list,
    grouped_seg_ids: list,
    avg_event: np.ndarray,
    fetching_fn,
) -> float:
    """Monotonicity using per-event SVs re-grouped by a reference segmentation.

    Per-event SHAP values are averaged within each group defined by
    *grouped_seg_ids* (e.g., transition-based change points), producing one
    aggregated SV per group.  Monotonicity is then evaluated at the group
    level using those aggregated SVs.

    Parameters
    ----------
    case : (1, L, D) ndarray
    y_hat : scalar – original model prediction probability
    per_event_sv : (n_events,) ndarray – one SHAP value per event (timestep)
    per_event_seg_ids : list of lists – each inner list has exactly one index,
        i.e. [[0], [1], [2], ...] as produced by the per-event strategy
    grouped_seg_ids : list of lists – reference grouping of timesteps,
        e.g. from the transition-based segmentation
    avg_event : (1, D) ndarray – perturbation baseline
    fetching_fn : wrapped model; f(x) -> (predictions (B,1), hidden_state)

    Returns
    -------
    float in [0, 1] – monotonicity score based on the grouped attributions
    """
    if not grouped_seg_ids:
        return 1.0

    # Build timestep → per-event SV mapping
    sv_flat = per_event_sv.flatten()
    t_to_sv: dict = {}
    for sv_val, seg in zip(sv_flat, per_event_seg_ids):
        for t in seg:
            t_to_sv[t] = float(sv_val)

    # Average per-event SVs within each group → one aggregated SV per group
    grouped_sv = np.array([
        np.mean([t_to_sv.get(t, 0.0) for t in seg])
        for seg in grouped_seg_ids
    ])

    return compute_monotonicity_for_case(
        case, y_hat, grouped_sv, grouped_seg_ids, avg_event, fetching_fn
    )


def compute_positive_relevance_monotonicity(
    case: np.ndarray,
    y_hat: float,
    segment_sv: np.ndarray,
    segment_ids: list,
    avg_event: np.ndarray,
    fetching_fn,
) -> float:
    """Positive-Relevance Monotonicity via Progressive Corruption.

    Only segments with positive SHAP attribution are considered.  They are
    sorted in descending order of their SHAP value (π) and then removed one
    by one from the trace (replaced by *avg_event*).  The cumulative score
    drop after each removal is Δ_i = f(σ) − f(σ^(i)), where σ^(i) is the
    trace with the first i positive-attribution segments zeroed out.

    The score counts the fraction of consecutive pairs (Δ_i, Δ_{i+1}) where
    the drop strictly increases:

        M⁺(σ) = 1/(m⁺−1) · Σ_{i=1}^{m⁺−1} 𝟙[Δ_{i+1} > Δ_i]

    A score of 1.0 means every successive removal of a positive-attribution
    segment further decreased the predicted probability, confirming that the
    SHAP ranking correctly orders segments by contribution.

    Parameters
    ----------
    case : (1, L, D) ndarray
    y_hat : scalar – original model prediction probability
    segment_sv : (n_segments,) ndarray – segment SHAP values
    segment_ids : list of lists – timestep indices per segment
    avg_event : (1, D) ndarray – perturbation baseline
    fetching_fn : wrapped model; f(x) -> (predictions (B,1), hidden_state)

    Returns
    -------
    float in [0, 1], or np.nan when fewer than 2 positive-attribution segments
    exist (score is undefined).
    """
    sv_flat = segment_sv.flatten()

    # R+ = segments with positive attribution
    pos_indices = [j for j, v in enumerate(sv_flat) if v > 0]
    m_pos = len(pos_indices)

    if m_pos < 2:
        return np.nan  # undefined: need at least 2 positive segments

    # π = sort positive segments by descending SHAP value
    pi = sorted(pos_indices, key=lambda j: sv_flat[j], reverse=True)

    # Compute Δ_i = f(σ) - f(σ^(i)) for i = 0, 1, ..., m+
    # Δ_0 = 0 by definition; each σ^(i) has the first i segments in π masked
    delta = [0.0]
    current = case.copy()
    for seg_idx in pi:
        for t in segment_ids[seg_idx]:
            current[0, t, :] = avg_event[0, :]
        pred, _ = fetching_fn(current)
        delta.append(y_hat - float(pred[0, 0]))

    # M+(σ) = 1/(m+-1) * Σ_{i=1}^{m+-1} 𝟙[Δ_{i+1} > Δ_i]
    count = sum(1 for i in range(1, m_pos) if delta[i + 1] > delta[i])
    return count / (m_pos - 1)


def compute_rpc_for_case(
    case: np.ndarray,
    y_hat: float,
    segment_sv: np.ndarray,
    segment_ids: list,
    avg_event: np.ndarray,
    fetching_fn,
    k: int,
) -> Tuple[float, float]:
    """Compute PGI and PGU for a single instance.

    PGI: perturb top-k segments (most important) → large gap expected.
    PGU: perturb non-top-k segments (least important) → small gap expected.

    Parameters
    ----------
    case : (1, L, D) ndarray
    y_hat : scalar – original model prediction probability
    segment_sv : (n_segments,) ndarray – segment SHAP values
    segment_ids : list of lists – timestep indices per segment
    avg_event : (1, D) ndarray – perturbation baseline
    fetching_fn : wrapped model; f(x) -> (predictions (B,1), hidden_state)
    k : number of top-important segments

    Returns
    -------
    pgi, pgu : floats
    """
    n_segments = len(segment_ids)
    k = min(k, n_segments)

    ranked = np.argsort(np.abs(segment_sv.flatten()))[::-1]
    top_k = ranked[:k].tolist()
    non_top_k = ranked[k:].tolist()

    if top_k:
        pred_pgi, _ = fetching_fn(
            _perturb_segments(case, segment_ids, top_k, avg_event)
        )
        pgi = float(abs(y_hat - float(pred_pgi[0, 0])))
        # relative prediction change after perturbation of important segments
        rpc_i = pgi / y_hat
    else:
        rpc_i = 0.0

    if non_top_k:
        pred_pgu, _ = fetching_fn(
            _perturb_segments(case, segment_ids, non_top_k, avg_event)
        )
        pgu = float(abs(y_hat - float(pred_pgu[0, 0])))
        # relative prediction change after perturbation of important segments
        rpc_u = pgu / y_hat
    else:
        rpc_u = 0.0

    return rpc_i, rpc_u


def compute_adaptive_k(
    explicands_info: Dict,
    coverage: float = 0.4,
) -> Dict[str, List[int]]:
    """For each cohort, find the minimum k such that the top-k segments
    (ranked by |SHAP value|) cover *coverage* fraction of the trace length
    on average across all cases in that cohort.

    Parameters
    ----------
    explicands_info : same nested dict used by evaluate_pgi_pgu
    coverage : target fraction of timesteps to cover (default 0.4 = 40%)

    Returns
    -------
    Dict[str, List[int]]  –  cohort → [k]   (single-element list so it can be
                              passed directly to evaluate_pgi_pgu)
    """
    k_per_cohort: Dict[str, List[int]] = {}

    for cohort, samples in explicands_info.items():
        k_values_per_case: List[int] = []

        for data in samples.values():
            for sv_info in data["sv"]:
                seg_ids = sv_info["segment_ids"]
                seg_sv = sv_info["segment_sv"].flatten()

                seg_lengths = np.array([len(s) for s in seg_ids])
                total_len = int(seg_lengths.sum())
                target = coverage * total_len

                ranked = np.argsort(np.abs(seg_sv))[::-1]
                cumulative = 0
                k = len(ranked)  # fallback: all segments needed
                for step, idx in enumerate(ranked, start=1):
                    cumulative += seg_lengths[idx]
                    if cumulative >= target:
                        k = step
                        break
                k_values_per_case.append(k)

        adaptive_k = int(np.ceil(np.mean(k_values_per_case)))
        k_per_cohort[cohort] = [adaptive_k]

        print(
            f"  cohort={cohort:6s}  "
            f"mean_k={np.mean(k_values_per_case):.2f}  "
            f"adaptive_k={adaptive_k}  "
            f"(coverage target={coverage:.0%}, n={len(k_values_per_case)} cases)"
        )

    return k_per_cohort


def evaluate_rpc(
    explicands_info: Dict,
    avg_event: np.ndarray,
    fetching_fn,
    k_values_per_cohort: Dict[str, List[int]],
    n_workers: int = 1,
    ref_explicands_info: "Dict | None" = None,
    compute_mono: bool = True,
) -> Dict:
    """Compute mean RPCi, RPCu, and Monotonicity for every cohort × predicted-class × k.

    Parameters
    ----------
    explicands_info : nested dict
        explicands_info[cohort][sample_name] must contain:
            "cases"  – list of (1, L, D) ndarrays
            "y_pred" – list of scalar prediction probabilities
            "sv"     – list of dicts with "segment_sv" and "segment_ids"
    k_values_per_cohort : mapping from cohort name to the list of k values
        to evaluate for that cohort, e.g.
        {"short": [1, 2], "medium": [1, 2, 3], "long": [1, 2, 3, 4, 5]}
    n_workers : number of threads for parallel per-case metric computation.
        Uses ThreadPoolExecutor — safe for model inference since PyTorch
        releases the GIL during C++ operations.
    ref_explicands_info : optional nested dict with the same structure as
        explicands_info.  When provided, the segment_ids from the reference
        strategy are used to group per-event SVs and a second monotonicity
        score ("grouped_mono") is computed and stored alongside the standard
        one.  Cases must be aligned (same order) with explicands_info.
    compute_mono : if False, skip compute_monotonicity_for_case and
        compute_monotonicity_grouped_for_case entirely; neither "mono" nor
        "grouped_mono" keys will appear in the results.

    Returns
    -------
    results[cohort][group][k] -> dict with
        rpc_i, rpc_u, rpc_i_std, rpc_u_std, n,
        and optionally mono, mono_std when compute_mono=True,
        pos_mono, pos_mono_std (when >= 2 positive-attribution segments exist),
        and grouped_mono, grouped_mono_std when ref_explicands_info is provided.
        group is one of: sample name (tp/fp/fn/tn),
                         "predicted_positive", "predicted_negative"
    """
    # tp/fp → predicted positive;  tn/fn → predicted negative
    PRED_CLASS = {
        "tp": "predicted_positive",
        "fp": "predicted_positive",
        "tn": "predicted_negative",
        "fn": "predicted_negative",
    }

    results: Dict = {}
    for cohort, samples in explicands_info.items():
        k_values = k_values_per_cohort.get(cohort, [1, 2, 3])

        def _empty_acc():
            return {k: [] for k in k_values}

        has_ref = ref_explicands_info is not None

        def _aggregate(acc_i, acc_u, acc_mono=None, acc_pos_mono=None, acc_grouped_mono=None):
            out = {}
            for k in k_values:
                d = {
                    "rpc_i": float(np.mean(acc_i[k])),
                    "rpc_u": float(np.mean(acc_u[k])),
                    "rpc_i_std": float(np.std(acc_i[k])),
                    "rpc_u_std": float(np.std(acc_u[k])),
                    "n": len(acc_i[k]),
                }
                if acc_mono is not None and acc_mono[k]:
                    d["mono"] = float(np.mean(acc_mono[k]))
                    d["mono_std"] = float(np.std(acc_mono[k]))
                if acc_pos_mono is not None:
                    valid = [v for v in acc_pos_mono[k] if not np.isnan(v)]
                    if valid:
                        d["pos_mono"] = float(np.mean(valid))
                        d["pos_mono_std"] = float(np.std(valid))
                if acc_grouped_mono is not None and acc_grouped_mono[k]:
                    d["grouped_mono"] = float(np.mean(acc_grouped_mono[k]))
                    d["grouped_mono_std"] = float(np.std(acc_grouped_mono[k]))
                out[k] = d
            return out

        results[cohort] = {}
        class_rpc_i = {
            "predicted_positive": _empty_acc(),
            "predicted_negative": _empty_acc(),
        }
        class_rpc_u = {
            "predicted_positive": _empty_acc(),
            "predicted_negative": _empty_acc(),
        }
        class_mono = {
            "predicted_positive": _empty_acc(),
            "predicted_negative": _empty_acc(),
        }
        class_pos_mono = {
            "predicted_positive": _empty_acc(),
            "predicted_negative": _empty_acc(),
        }
        class_grouped_mono = (
            {"predicted_positive": _empty_acc(), "predicted_negative": _empty_acc()}
            if has_ref else None
        )

        for name, data in samples.items():
            results[cohort][name] = {}
            pred_class = PRED_CLASS[name]

            # Monotonicity is independent of k — compute once per case (parallel)
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm

            ref_data = (
                ref_explicands_info.get(cohort, {}).get(name)
                if has_ref else None
            )
            _args = [
                (
                    case,
                    y_hat,
                    sv_info["segment_sv"],
                    sv_info["segment_ids"],
                    ref_sv_info["segment_ids"] if ref_sv_info is not None else None,
                )
                for case, y_hat, sv_info, ref_sv_info in zip(
                    data["cases"],
                    data["y_pred"],
                    data["sv"],
                    ref_data["sv"] if ref_data is not None else [None] * len(data["cases"]),
                )
            ]

            def _compute_all(a):
                case_, y_hat_, sv_, seg_ids_, ref_seg_ids_ = a
                mono = (
                    compute_monotonicity_for_case(case_, y_hat_, sv_, seg_ids_, avg_event, fetching_fn)
                    if compute_mono else None
                )
                pos_mono = compute_positive_relevance_monotonicity(case_, y_hat_, sv_, seg_ids_, avg_event, fetching_fn)
                grouped_mono = (
                    compute_monotonicity_grouped_for_case(
                        case_, y_hat_, sv_, seg_ids_, ref_seg_ids_, avg_event, fetching_fn
                    )
                    if (compute_mono and ref_seg_ids_ is not None) else None
                )
                return mono, pos_mono, grouped_mono

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                all_list = list(
                    tqdm(
                        executor.map(_compute_all, _args),
                        total=len(_args),
                        desc=f"Mono [{cohort}/{name}]",
                    )
                )
            mono_list = [v[0] for v in all_list]       # None entries when compute_mono=False
            pos_mono_list = [v[1] for v in all_list]   # np.nan when < 2 positive segments
            grouped_mono_list = [v[2] for v in all_list]  # None entries when no ref or compute_mono=False

            for k in k_values:
                rpc_i_list, rpc_u_list = [], []
                for case, y_hat, sv_info in zip(
                    data["cases"], data["y_pred"], data["sv"]
                ):
                    rpc_i, rpc_u = compute_rpc_for_case(
                        case,
                        y_hat,
                        sv_info["segment_sv"],
                        sv_info["segment_ids"],
                        avg_event,
                        fetching_fn,
                        k,
                    )
                    rpc_i_list.append(rpc_i)
                    rpc_u_list.append(rpc_u)

                valid_mono = [v for v in mono_list if v is not None]
                valid_pos_mono = [v for v in pos_mono_list if not np.isnan(v)]
                valid_grouped = [v for v in grouped_mono_list if v is not None]
                entry = {
                    "rpc_i": float(np.mean(rpc_i_list)),
                    "rpc_u": float(np.mean(rpc_u_list)),
                    "rpc_i_std": float(np.std(rpc_i_list)),
                    "rpc_u_std": float(np.std(rpc_u_list)),
                    "n": len(rpc_i_list),
                }
                if valid_mono:
                    entry["mono"] = float(np.mean(valid_mono))
                    entry["mono_std"] = float(np.std(valid_mono))
                if valid_pos_mono:
                    entry["pos_mono"] = float(np.mean(valid_pos_mono))
                    entry["pos_mono_std"] = float(np.std(valid_pos_mono))
                if valid_grouped:
                    entry["grouped_mono"] = float(np.mean(valid_grouped))
                    entry["grouped_mono_std"] = float(np.std(valid_grouped))
                results[cohort][name][k] = entry

                class_rpc_i[pred_class][k].extend(rpc_i_list)
                class_rpc_u[pred_class][k].extend(rpc_u_list)
                class_mono[pred_class][k].extend(valid_mono)
                class_pos_mono[pred_class][k].extend(pos_mono_list)
                if class_grouped_mono is not None:
                    class_grouped_mono[pred_class][k].extend(valid_grouped)

        results[cohort]["predicted_positive"] = _aggregate(
            class_rpc_i["predicted_positive"],
            class_rpc_u["predicted_positive"],
            class_mono["predicted_positive"],
            class_pos_mono["predicted_positive"],
            class_grouped_mono["predicted_positive"] if class_grouped_mono else None,
        )
        results[cohort]["predicted_negative"] = _aggregate(
            class_rpc_i["predicted_negative"],
            class_rpc_u["predicted_negative"],
            class_mono["predicted_negative"],
            class_pos_mono["predicted_negative"],
            class_grouped_mono["predicted_negative"] if class_grouped_mono else None,
        )

    return results


def save_rpc_latex_table(
    results_per_strategy: Dict[str, Dict], output_path: str = ""
) -> str:
    r"""Render RPC results for multiple segmentation strategies as a LaTeX table.

    One ``{table}`` environment is emitted per (cohort × k) combination.
    Strategy blocks are stacked vertically, each sharing the same row labels.
    Uses \hline and \resizebox{\columnwidth} formatting.

    Parameters
    ----------
    results_per_strategy : Dict[str, Dict]
        Mapping from strategy name to the return value of ``evaluate_rpc``.
    output_path : if non-empty the LaTeX string is written to this file

    Returns
    -------
    str  – the complete LaTeX source
    """
    lines: List[str] = []

    # Collect all cohorts across all strategies
    cohorts: set = set()
    for results in results_per_strategy.values():
        cohorts.update(results.keys())

    # Detect whether any strategy has pos_mono / grouped_mono results
    def _any_key(key):
        return any(
            key in m
            for results in results_per_strategy.values()
            for cohort_data in results.values()
            for group_data in cohort_data.values()
            for m in (group_data.values() if isinstance(group_data, dict) else [])
            if isinstance(m, dict)
        )

    has_pos_mono = _any_key("pos_mono")
    n_cols = 3 + int(has_pos_mono) + 1  # label + RPCi + RPCu + optional pos_mono + k
    col_spec = "l" + " c" * (n_cols - 1)

    for cohort in sorted(cohorts):
        header = (
            r"\textbf{Sample}"
            r" & $\mathrm{RPCI}$"
            r" & $\mathrm{RPCU}$"
        )
        if has_pos_mono:
            header += r" & Pos. Relevance Mono"
        header += r" & $k$ \\ \midrule"

        lines += [
            r"\begin{table}[!ht]",
            r"\centering",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
        ]

        for strategy, results in results_per_strategy.items():
            cohort_data = results.get(cohort, {})

            strategy_label = _STRATEGY_LABELS.get(strategy, strategy)
            lines.append(
                rf"\multicolumn{{{n_cols}}}{{l}}{{\mathrm{{{strategy_label}}}}} \\ \midrule"
            )

            # Split groups: individual samples vs aggregated predicted classes
            present_groups = [g for g in _ROW_ORDER if g in cohort_data]
            sample_groups = [g for g in present_groups if g in SAMPLE_NAMES]
            class_groups = [g for g in present_groups if g not in SAMPLE_NAMES]

            # Gather all k values
            k_vals = sorted({k for g in present_groups for k in cohort_data[g]})

            for k in k_vals:
                # ── Individual sample rows (tp / fp / fn / tn) ──
                for group in sample_groups:
                    if k not in cohort_data[group]:
                        continue
                    m = cohort_data[group][k]
                    label = _ROW_LABELS.get(group, group.upper())
                    rpc_i = f"{m['rpc_i']:.3f} $\\pm$ {m['rpc_i_std']:.3f}"
                    rpc_u = f"{m['rpc_u']:.3f} $\\pm$ {m['rpc_u_std']:.3f}"
                    row = rf"{label} & {rpc_i} & {rpc_u}"
                    if has_pos_mono:
                        pm_cell = (
                            f"${m['pos_mono']:.3f} \\pm {m.get('pos_mono_std', 0.0):.3f}$"
                            if "pos_mono" in m else "---"
                        )
                        row += f" & {pm_cell}"
                    row += rf" & {k} \\"
                    lines.append(row)

                # ── Separator before aggregated class rows ──
                if sample_groups and class_groups:
                    lines.append(r"\midrule")

                # ── Aggregated predicted-class rows ──
                for group in class_groups:
                    if k not in cohort_data[group]:
                        continue
                    m = cohort_data[group][k]
                    label = _ROW_LABELS.get(group, group.upper())
                    rpc_i = f"{m['rpc_i']:.3f} $\\pm$ {m['rpc_i_std']:.3f}"
                    rpc_u = f"{m['rpc_u']:.3f} $\\pm$ {m['rpc_u_std']:.3f}"
                    row = rf"{label} & {rpc_i} & {rpc_u}"
                    if has_pos_mono:
                        pm_cell = (
                            f"${m['pos_mono']:.3f} \\pm {m.get('pos_mono_std', 0.0):.3f}$"
                            if "pos_mono" in m else "---"
                        )
                        row += f" & {pm_cell}"
                    row += rf" & {k} \\"
                    lines.append(row)

            lines.append(r"\bottomrule")

        lines += [
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]

    tex = "\n".join(lines) + "\n"

    if output_path:
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(tex)
        print(f"Saved LaTeX table → {output_path}")

    return tex


def print_rpc_table(results: Dict, output_path: str = "", mono_key: "str | None" = None) -> None:
    """Print (and optionally save) an RPC results table.

    Parameters
    ----------
    results : return value of evaluate_rpc / aggregate_random_results
    output_path : if non-empty, write the table to this CSV path
    mono_key : optional key from the metric dict to include as a "Mono" column,
        e.g. ``"mono"`` (ungrouped) or ``"grouped_mono"`` (grouped per-event).
        When None no additional mono column is added.
    """
    mono_label = (
        "Grouped Mono" if mono_key == "grouped_mono" else "Mono"
    ) if mono_key else None

    rows = []
    for cohort, samples in results.items():
        for name, k_dict in samples.items():
            for k, m in k_dict.items():
                pm_mean = round(m["pos_mono"], 4) if "pos_mono" in m else None
                pm_std = round(m.get("pos_mono_std", 0.0), 4) if "pos_mono" in m else None
                row = {
                    "cohort": cohort,
                    "sample": name.upper(),
                    "k": k,
                    "RPCi (mean)": round(m["rpc_i"], 4),
                    "RPCi (std)": round(m["rpc_i_std"], 4),
                    "RPCu (mean)": round(m["rpc_u"], 4),
                    "RPCu (std)": round(m["rpc_u_std"], 4),
                    "Pos Mono (mean)": pm_mean,
                    "Pos Mono (std)": pm_std,
                    "n": m["n"],
                }
                if mono_key is not None:
                    mono_mean = round(m[mono_key], 4) if mono_key in m else None
                    mono_std = round(m.get(f"{mono_key}_std", 0.0), 4) if mono_key in m else None
                    row[f"{mono_label} (mean)"] = mono_mean
                    row[f"{mono_label} (std)"] = mono_std
                rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if output_path:
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved table → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def aggregate_random_results(results_per_seed: List[Dict]) -> Dict:
    """Average evaluate_rpc metric dicts across multiple random seeds.

    Each seed was run on the same set of instances (same n), so means are
    averaged uniformly and the combined std accounts for both within-seed
    variance and between-seed variance of the mean:

        pooled_std = sqrt( mean(std_i²) + mean((mean_i − pooled_mean)²) )

    n is taken from the first seed (identical across seeds).

    Parameters
    ----------
    results_per_seed : list of evaluate_rpc return dicts, one per seed

    Returns
    -------
    Dict with the same structure as a single evaluate_rpc result.
    """
    _SCALAR_KEYS = ["rpc_i", "rpc_u", "mono", "pos_mono", "grouped_mono"]

    if not results_per_seed:
        return {}

    ref = results_per_seed[0]
    aggregated: Dict = {}

    for cohort, cohort_data in ref.items():
        aggregated[cohort] = {}
        for group, group_data in cohort_data.items():
            aggregated[cohort][group] = {}
            for k in group_data:
                seed_entries = [
                    r[cohort][group][k]
                    for r in results_per_seed
                    if cohort in r and group in r[cohort] and k in r[cohort][group]
                ]
                if not seed_entries:
                    continue

                entry: Dict = {"n": seed_entries[0]["n"]}

                for key in _SCALAR_KEYS:
                    std_key = f"{key}_std"
                    valid = [e for e in seed_entries if key in e]
                    if not valid:
                        continue
                    means = [e[key] for e in valid]
                    vars_ = [e[std_key] ** 2 for e in valid]
                    pooled_mean = sum(means) / len(means)
                    pooled_var = (
                        sum(vars_) / len(vars_)
                        + sum((m - pooled_mean) ** 2 for m in means) / len(means)
                    )
                    entry[key] = float(pooled_mean)
                    entry[std_key] = float(np.sqrt(max(pooled_var, 0.0)))

                aggregated[cohort][group][k] = entry

    return aggregated


def load_seed_explicands(sv_dir: str, cohort: str, seed: int) -> Dict[str, Dict]:
    """Load random-segmentation results for a single seed.

    Parameters
    ----------
    sv_dir : root directory containing per-seed folders
    cohort : cohort name, e.g. "medium"
    seed   : integer seed value

    Returns
    -------
    Dict[sample_name -> loaded pickle data]
    """
    seed_dir = osp.join(sv_dir, f"random_cohort_{cohort}_seed{seed}")
    data: Dict[str, Dict] = {}
    for name in SAMPLE_NAMES:
        pkl_path = osp.join(seed_dir, f"{name}_segment_sv_results.pkl")
        with open(pkl_path, "rb") as f:
            data[name] = pickle.load(f)
    return data


def main():
    # ── 1. Load saved SHAP explanations per cohort × strategy ───────
    explicands_per_strategy: Dict[str, Dict[str, Dict]] = {}

    print("=" * 80)
    print("Loading SHAP explanations and segments info.")
    print("=" * 80)

    for strategy in SEG_STRATEGIES:
        if strategy == "random":
            continue  # handled per-seed during evaluation
        explicands_per_strategy[strategy] = {}
        for cohort in COHORT_ORDER:
            sv_cohort_dir = osp.join(sv_output_dir, f"{strategy}_cohort_{cohort}")
            explicands_per_strategy[strategy][cohort] = {}
            for name in SAMPLE_NAMES:
                pkl_path = osp.join(sv_cohort_dir, f"{name}_segment_sv_results.pkl")
                with open(pkl_path, "rb") as f:
                    explicands_per_strategy[strategy][cohort][name] = pickle.load(f)

    # ── 2. Load model and build average-event baseline ───────────────
    print("\nLoading model and building average-event baseline")
    config, train_loader, _, model = load_data_and_model(config_path, checkpoint_path)
    avg_event, _ = build_average_event_baseline(train_loader, config)
    avg_event = np.asarray(avg_event)

    wrapped_model = OutcomePredictorWrapper(
        model, batch_budget=1, categorical_indices=[0], device=config["device"]
    )

    def fetching_fn(x, hs=None):
        return wrapped_model(sequences=x, hidden_state=hs)

    # ── 3. Compute RPCi / RPCu per strategy ────────────────────────
    rpc_per_strategy: Dict[str, Dict] = {}
    print("=" * 80)
    print("Computing RPCi / RPCu per strategy")
    print("=" * 80)

    for strategy, explicands_info in explicands_per_strategy.items():
        print(f"--- Strategy: {strategy} ---")
        k_per_cohort = compute_adaptive_k(explicands_info, coverage=0.4)

        # For per-event strategy, also compute grouped monotonicity using
        # transition-based change points to re-group the per-event SVs.
        ref_info = (
            explicands_per_strategy.get("transition")
            if strategy == "per_event"
            else None
        )

        print(
            f"Evaluating RPCi / RPCu for k in {k_per_cohort} "
            f"perturbing with the average event as a baseline..."
        )
        rpc_per_strategy[strategy] = evaluate_rpc(
            explicands_info,
            avg_event,
            fetching_fn,
            k_values_per_cohort=k_per_cohort,
            n_workers=9,
            ref_explicands_info=ref_info,
            compute_mono=(strategy == "per_event"),
        )

    # ── Random strategy: evaluate per seed, then aggregate ──────────
    print("--- Strategy: random ---")
    seed_results: List[Dict] = []
    k_per_cohort_random: Dict = {}
    for seed in RANDOM_SEEDS:
        seed_explicands = {
            cohort: load_seed_explicands(sv_output_dir, cohort, seed)
            for cohort in COHORT_ORDER
        }
        if not k_per_cohort_random:
            k_per_cohort_random = compute_adaptive_k(seed_explicands, coverage=0.4)
        print(
            f"  seed={seed}: evaluating RPCi / RPCu for k in {k_per_cohort_random} ..."
        )
        seed_results.append(evaluate_rpc(
            seed_explicands,
            avg_event,
            fetching_fn,
            k_values_per_cohort=k_per_cohort_random,
            n_workers=9,
            ref_explicands_info=None,
            compute_mono=False,
        ))
    rpc_per_strategy["random"] = aggregate_random_results(seed_results)

    # ── 4. Report ────────────────────────────────────────────────────
    report_order = ["per_event", "distribution", "transition"]
    for strategy in report_order:
        if strategy not in rpc_per_strategy:
            continue
        print(f"\n--- RPCi / RPCu results [{strategy}] (avg_event baseline) ---")
        csv_path = osp.join(sv_output_dir, f"rpc_table_{strategy}.csv")
        mono_key = (
            ("grouped_mono" if USE_GROUPED_MONO_FOR_PER_EVENT else "mono")
            if strategy == "per_event"
            else None
        )
        print_rpc_table(rpc_per_strategy[strategy], output_path=csv_path, mono_key=mono_key)

    ordered_rpc = {s: rpc_per_strategy[s] for s in report_order if s in rpc_per_strategy}
    tex_path = osp.join(OUTPUT_ROOT, "shap_values", "bpi17", "rpc_table.tex")
    save_rpc_latex_table(ordered_rpc, output_path=tex_path)


if __name__ == "__main__":
    main()
