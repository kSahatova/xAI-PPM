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
COHORT_ORDER = ["medium"]
SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]
TOP_N_PATTERNS = 10

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


_ROW_ORDER = ["predicted_positive", "predicted_negative"]  # "tp", "fp", "fn", "tn",
_ROW_LABELS = {
    # "tp": "TP (cancelled, correct)",
    # "fp": "FP (accepted, wrong)",
    # "fn": "FN (cancelled, wrong)",
    # "tn": "TN (accepted, correct)",
    "predicted_positive": "Predicted positive",
    "predicted_negative": "Predicted negative",
}
_STRATEGY_LABELS = {
    "per_event": "Per-event segmentation",
    # "random": "Random segmentation",
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
    epsilon: float = 1e-4,
) -> float:
    """Compute the monotonicity score for a single instance.

    Starts from a fully masked trace and incrementally restores segments in
    descending order of |SHAP| importance.  At each step the distance to the
    original prediction should decrease — the fraction of steps where this
    holds (within tolerance ε) is the monotonicity score.

    Parameters
    ----------
    case : (1, L, D) ndarray
    y_hat : scalar – original model prediction probability
    segment_sv : (n_segments,) ndarray – segment SHAP values
    segment_ids : list of lists – timestep indices per segment
    avg_event : (1, D) ndarray – perturbation baseline
    fetching_fn : wrapped model; f(x) -> (predictions (B,1), hidden_state)
    epsilon : tolerance added to each step's distance threshold

    Returns
    -------
    float in [0, 1] – fraction of monotonically improving steps
    """
    n_segments = len(segment_ids)
    if n_segments == 0:
        return 1.0

    # Step 1: fully masked trace
    masked = _perturb_segments(case, segment_ids, list(range(n_segments)), avg_event)

    # Step 2: rank segments by descending |SHAP|
    order = np.argsort(np.abs(segment_sv.flatten()))[::-1].tolist()

    # Step 3: incremental restoration, collecting one prediction per step
    pred_masked, _ = fetching_fn(masked)
    predictions = [float(pred_masked[0, 0])]
    current = masked.copy()
    for seg_idx in order:
        for t in segment_ids[seg_idx]:
            current[0, t, :] = case[0, t, :]
        pred, _ = fetching_fn(current)
        predictions.append(float(pred[0, 0]))

    # Step 4: distance to original prediction at each step
    distances = [abs(y_hat - p) for p in predictions]

    # Step 5: count steps where distance decreased (within epsilon)
    monotonic_steps = sum(
        1 for i in range(1, len(distances)) if distances[i] < distances[i - 1] + epsilon
    )
    return monotonic_steps / (len(distances) - 1)


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
    epsilon: float = 1e-4,
    n_workers: int = 1,
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
    epsilon : tolerance for monotonicity step check
    n_workers : number of threads for parallel monotonicity computation.
        Uses ThreadPoolExecutor — safe for model inference since PyTorch
        releases the GIL during C++ operations.

    Returns
    -------
    results[cohort][group][k] -> dict with
        rpc_i, rpc_u, rpc_i_std, rpc_u_std, mono, mono_std, n
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

        def _aggregate(acc_i, acc_u, acc_mono):
            return {
                k: {
                    "rpc_i": float(np.mean(acc_i[k])),
                    "rpc_u": float(np.mean(acc_u[k])),
                    "rpc_i_std": float(np.std(acc_i[k])),
                    "rpc_u_std": float(np.std(acc_u[k])),
                    "mono": float(np.mean(acc_mono[k])),
                    "mono_std": float(np.std(acc_mono[k])),
                    "n": len(acc_i[k]),
                }
                for k in k_values
            }

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

        for name, data in samples.items():
            results[cohort][name] = {}
            pred_class = PRED_CLASS[name]

            # Monotonicity is independent of k — compute once per case (parallel)
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm

            _args = [
                (case, y_hat, sv_info["segment_sv"], sv_info["segment_ids"])
                for case, y_hat, sv_info in zip(
                    data["cases"], data["y_pred"], data["sv"]
                )
            ]
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                mono_list = list(
                    tqdm(
                        executor.map(
                            lambda a: compute_monotonicity_for_case(
                                a[0], a[1], a[2], a[3], avg_event, fetching_fn, epsilon
                            ),
                            _args,
                        ),
                        total=len(_args),
                        desc=f"Monotonicity [{cohort}/{name}]",
                    )
                )

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
                results[cohort][name][k] = {
                    "rpc_i": float(np.mean(rpc_i_list)),
                    "rpc_u": float(np.mean(rpc_u_list)),
                    "rpc_i_std": float(np.std(rpc_i_list)),
                    "rpc_u_std": float(np.std(rpc_u_list)),
                    "mono": float(np.mean(mono_list)),
                    "mono_std": float(np.std(mono_list)),
                    "n": len(rpc_i_list),
                }
                class_rpc_i[pred_class][k].extend(rpc_i_list)
                class_rpc_u[pred_class][k].extend(rpc_u_list)
                class_mono[pred_class][k].extend(mono_list)

        results[cohort]["predicted_positive"] = _aggregate(
            class_rpc_i["predicted_positive"],
            class_rpc_u["predicted_positive"],
            class_mono["predicted_positive"],
        )
        results[cohort]["predicted_negative"] = _aggregate(
            class_rpc_i["predicted_negative"],
            class_rpc_u["predicted_negative"],
            class_mono["predicted_negative"],
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

    def _pooled_mono(group_metrics: List[Dict]) -> Tuple[float, float]:
        """Pooled mean and std of monotonicity across multiple groups."""
        total_n = sum(m["n"] for m in group_metrics)
        if total_n == 0:
            return 0.0, 0.0
        pooled_mean = sum(m["mono"] * m["n"] for m in group_metrics) / total_n
        pooled_var = (
            sum(
                m["n"] * (m["mono_std"] ** 2 + (m["mono"] - pooled_mean) ** 2)
                for m in group_metrics
            )
            / total_n
        )
        return pooled_mean, float(np.sqrt(max(pooled_var, 0.0)))

    for cohort in sorted(cohorts):
        lines += [
            r"\begin{table}[!ht]",
            r"\centering",
            r"\begin{tabular}{l c c c c}",
            r"\toprule",
            r"\textbf{Predicted class}"
            r" & $\mathrm{RPCI}$"
            r" & $\mathrm{RPCU}$"
            r" & Monotonicity"
            r" & $k$ \\ \midrule",
        ]

        for strategy, results in results_per_strategy.items():
            cohort_data = results.get(cohort, {})

            strategy_label = _STRATEGY_LABELS.get(strategy, strategy)
            lines.append(
                rf"\multicolumn{{5}}{{l}}{{\mathrm{{{strategy_label}}}}} \\ \midrule"
            )

            # Determine which groups are present and collect rows per k
            present_groups = [g for g in _ROW_ORDER if g in cohort_data]
            n_groups = len(present_groups)

            # Gather all k values
            k_vals = sorted({k for g in present_groups for k in cohort_data[g]})

            for k in k_vals:
                # Combined (pooled) monotonicity across all present groups for this k
                group_metrics_for_k = [
                    cohort_data[g][k] for g in present_groups if k in cohort_data[g]
                ]
                c_mean, c_std = _pooled_mono(group_metrics_for_k)
                mono_cell = (
                    rf"\multirow{{{n_groups}}}{{*}}{{${c_mean:.3f} \pm {c_std:.3f}$}}"
                )

                for row_idx, group in enumerate(present_groups):
                    if k not in cohort_data[group]:
                        continue
                    m = cohort_data[group][k]
                    label = _ROW_LABELS.get(group, group.upper())
                    rpc_i = f"{m['rpc_i']:.3f} $\\pm$ {m['rpc_i_std']:.3f}"
                    rpc_u = f"{m['rpc_u']:.3f} $\\pm$ {m['rpc_u_std']:.3f}"
                    # Monotonicity cell only in first row; empty in subsequent rows
                    mono_part = mono_cell if row_idx == 0 else ""
                    lines.append(rf"{label} & {rpc_i} & {rpc_u} & {mono_part} & {k} \\")

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


def print_rpc_table(results: Dict) -> None:
    rows = []
    for cohort, samples in results.items():
        for name, k_dict in samples.items():
            for k, m in k_dict.items():
                rows.append(
                    {
                        "cohort": cohort,
                        "sample": name.upper(),
                        "k": k,
                        "RPCi (mean)": round(m["rpc_i"], 4),
                        "RPCi (std)": round(m["rpc_i_std"], 4),
                        "RPCu (mean)": round(m["rpc_u"], 4),
                        "RPCu (std)": round(m["rpc_u_std"], 4),
                        "Mono (mean)": round(m["mono"], 4),
                        "Mono (std)": round(m["mono_std"], 4),
                        "n": m["n"],
                    }
                )

    print(pd.DataFrame(rows).to_string(index=False))


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    # ── 1. Load saved SHAP explanations per cohort × strategy ───────
    explicands_per_strategy: Dict[str, Dict[str, Dict]] = {}

    print("=" * 80)
    print("Loading SHAP explanations and segments info.")
    print("=" * 80)

    for strategy in SEG_STRATEGIES:
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
        k_per_cohort = compute_adaptive_k(explicands_info, coverage=0.3)

        print(
            f"Evaluating RPCi / RPCu for k in {k_per_cohort} "
            f"perturbing with the average event as a baseline..."
        )
        rpc_per_strategy[strategy] = evaluate_rpc(
            explicands_info,
            avg_event.to_numpy(),
            fetching_fn,
            k_values_per_cohort=k_per_cohort,
            n_workers=9,
        )

    # ── 4. Report ────────────────────────────────────────────────────
    for strategy, results in rpc_per_strategy.items():
        print(f"\n--- RPCi / RPCu results [{strategy}] (avg_event baseline) ---")
        print_rpc_table(results)

    tex_path = osp.join(OUTPUT_ROOT, "shap_values", "bpi17", "rpc_table.tex")
    save_rpc_latex_table(rpc_per_strategy, output_path=tex_path)

    # ── 5. Persist ───────────────────────────────────────────────────
    # out_path = osp.join(sv_output_dir, "rpci_rpcu_results_avg_pert_strategy.pkl")
    # with open(out_path, "wb") as f:
    #     pickle.dump(rpc_per_strategy, f)
    # print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
