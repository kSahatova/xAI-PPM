#!/usr/bin/env python
"""
Base experiment — Segment-level SHAP values via transition-based segmentation.

Pipeline
--------
1. Load data & pre-trained model
2. Extract TP / TN / FP / FN cases (single-offer only)
3. Build baselines (average-event & most-frequent-variant)
4. Segment each trace with the control-flow-aware PELT algorithm
5. Compute segment-level SHAP values for every explicand
6. Persist raw results to disk for downstream analyses

Outputs
-------
<OUTPUT_ROOT>/segmentation/bpi17/           — segmentation visualisations
<OUTPUT_ROOT>/shap_values/bpi17_transition/ — SHAP force / heatmap / bar plots
<OUTPUT_ROOT>/shap_values/bpi17_transition/segment_sv_results.npz — raw SV arrays

Usage
-----
    python 01_base_experiment.py
"""

import os
import pickle
import numpy as np
import os.path as osp
from typing import Dict

from experiments.setup_experiment import load_data_and_model
from ppm.utils import extract_explicands_samples
from local_xai.utils.baseline_calculation import build_average_event_baseline
from local_xai.utils.trace_segmentation.transition_based import build_transition_matrix
from local_xai.utils.trace_segmentation.transition_based import (
    calculate_and_plot_segments,
)
from timeshap.wrappers.outcome_predictor_wrapper import OutcomePredictorWrapper

from shap.utils._legacy import IdentityLink
from local_xai.seqshap import (
    SeqShapKernel,
    plot_feature_level_sv,
    plot_segment_level_sv,
)


SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]
PREFIX_LEN = 15
EXPLICANDS_NUM = 10

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")
config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
checkpoint_path = osp.join(PROJECT_DIR, r"persisted_models\suffix\BPI17_rnn_outcome_bpi17.pth")


def compute_segment_shap_values(
    fetching_f,
    explicands: Dict,
    segments: Dict[str, list],
    baseline: np.ndarray,
    config: dict,
    *,
    output_dir: str = "",
    save_plots: bool = True,
) -> Dict[str, list]:
    """Compute segment-level (and optionally feature-level) SHAP values.

    Returns
    -------
    results : dict
        sample_name → list of dicts with keys:
            segment_sv, feature_sv, segment_names, segment_explainer,
            feature_explainer, base_value
    """
    results: Dict[str, list] = {}

    for name in SAMPLE_NAMES:
        cases = explicands[name]["cases"]
        segs = segments[name]
        case_results = []

        if save_plots:
            os.makedirs(osp.join(output_dir, name), exist_ok=True)

        for i, (case, seg_info) in enumerate(zip(cases, segs)):
            seg_ids = seg_info["segment_ids"]

            # --- segment-level ---
            seg_explainer = SeqShapKernel(
                fetching_f,
                baseline,
                rs=52,
                mode="segment",
                segment_ids=seg_ids,
                link=IdentityLink(),
            )
            seg_sv = seg_explainer.shap_values(case)
            seg_names = [f"Segment:  {i + 1}" for i in range(seg_sv.shape[0])]

            if save_plots:
                plot_segment_level_sv(
                    seg_sv, seg_explainer, seg_names, name, i, output_dir
                )

            # --- feature-level ---
            feat_explainer = SeqShapKernel(fetching_f, baseline, rs=52, mode="feature")
            feat_sv = feat_explainer.shap_values(case)

            if save_plots:
                plot_feature_level_sv(
                    feat_sv, feat_explainer, config, name, i, output_dir
                )

            case_results.append(
                {
                    "segment_sv": seg_sv,
                    "feature_sv": feat_sv,
                    "segment_names": seg_names,
                    "segment_ids": seg_ids,
                    # "segment_explainer": seg_explainer,
                    # "feature_explainer": feat_explainer,
                    "base_value": seg_explainer.expected_value,
                }
            )

        results[name] = case_results
        print(f"  [{name.upper()}] computed SV for {len(case_results)} cases")

    return results


def main():
    # ── 1. Load data & model ────────────────────────────────────────
    print("Loading data and model …")
    config, train_loader, test_loader, model = load_data_and_model(
        config_path, checkpoint_path
    )
    # evaluate_model(model, test_loader, device=config["device"])

    # ── 2. Extract & filter cases ───────────────────────────────────
    print("\nExtracting prediction cases …")
    explicands_info = extract_explicands_samples(
        model, test_loader, PREFIX_LEN, EXPLICANDS_NUM
    )

    # ── 3. Baselines ────────────────────────────────────────────────
    print("\nBuilding baselines …")
    avg_event, avg_sequence = build_average_event_baseline(train_loader, config)
    # baseline_acc, baseline_canc = build_variant_baselines(train_loader, config)

    # # Choose which baseline to use for SV computation
    # baseline = baseline_acc[:, :PREFIX_LEN]

    # ── 4. Transition-based segmentation ────────────────────────────
    print("\nSegmenting traces (transition-based PELT) …")
    train_df = train_loader.dataset.log.dataframe
    unique_activities_num = train_df["activity"].nunique()
    transition_matrix = build_transition_matrix(train_df, unique_activities_num)

    activity_lookup = test_loader.dataset.log.itos["activity"]

    seg_output_dir = osp.join(OUTPUT_ROOT, "segmentation", "bpi17")
    segments_info = {}
    for name in SAMPLE_NAMES:
        sample_seg_output_dir  = osp.join(seg_output_dir, name)
        os.makedirs(sample_seg_output_dir, exist_ok=True)

        segments = calculate_and_plot_segments(
            explicands_info[name]["cases"],
            explicands_info[name]["predictions"],
            explicands_info[name]["labels"],
            transition_matrix,
            plot_flag=True,
            output_dir=sample_seg_output_dir,
            activity_lookup=activity_lookup,
        )
        segments_info[name] = segments

    # ── 5. Segment-level SHAP values ───────────────────────────────
    print("\nComputing segment-level SHAP values …")
    wrapped_model = OutcomePredictorWrapper(
        model, batch_budget=1, categorical_indices=[0], device=config["device"]
    )
    fetching_fn = lambda x, hs=None: wrapped_model(sequences=x, hidden_state=hs)

    sv_output_dir = osp.join(OUTPUT_ROOT, "shap_values", "bpi17_transition")
    sv_results = compute_segment_shap_values(
        fetching_fn,
        explicands_info,
        segments_info,
        avg_event,
        config,
        output_dir=sv_output_dir,
        save_plots=True,
    )

    # ── 6. Persist results for ablation / correlation scripts ──────
    print("\nSaving raw results …")
    results_path = osp.join(sv_output_dir, "segment_sv_results.pkl")
    payload = {
        "sv_results": sv_results,
        "segments": segments_info,
        "explicands": explicands_info,
        "baseline": avg_event,
        "config": config,
    }
    with open(results_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved to {results_path}")
    
    print("\n✓ Base experiment complete.")


if __name__ == "__main__":
    main()
