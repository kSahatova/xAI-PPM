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
import seaborn as sns
import os.path as osp
import colorcet as cc
from typing import Dict, List, Any
import matplotlib.pyplot as plt

from experiments.setup_experiment import load_data_and_model
from local_xai.utils.trace_segmentation.visualization import (
    plot_segmented_trace_vertical,
)
from ppm.utils import calculate_accuracy, extract_explicands_samples
from local_xai.utils.baseline_calculation import build_average_event_baseline
from local_xai.utils.trace_segmentation.transition_based import build_transition_matrix
from local_xai.utils.trace_segmentation.segmentation_factory import get_segmenter
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
checkpoint_path = osp.join(
    OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth"
)


def compute_segment_shap_values(
    fetching_f,
    explicands: List,
    segments: List[Dict[str, list]],
    baseline: np.ndarray,
    config: dict,
    *,
    output_dir: str = "",
    save_plots: bool = True,
) -> List[Dict[str, Any]]:
    """Compute segment-level (and optionally feature-level) SHAP values.

    Returns
    -------
    results : list of dicts
        sample_name → list of dicts with keys:
            segment_sv, feature_sv, segment_names, segment_explainer,
            feature_explainer, base_value
    """

    case_results = []

    if save_plots:
        os.makedirs(osp.join(output_dir), exist_ok=True)

    for i, (case, seg_info) in enumerate(zip(explicands, segments)):
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
                seg_sv, seg_explainer, seg_names, i, output_dir
            )

        # --- feature-level ---
        feat_explainer = SeqShapKernel(fetching_f, baseline, rs=52, mode="feature")
        feat_sv = feat_explainer.shap_values(case)

        if save_plots:
            plot_feature_level_sv(
                feat_sv, feat_explainer, config, i, output_dir
            )

        case_results.append(
            {
                "segment_sv": seg_sv,
                "feature_sv": feat_sv,
                "segment_names": seg_names,
                "segment_ids": seg_ids,
                "base_value": seg_explainer.expected_value,
            }
        )

    return case_results


def main():
    # ── 1. Load data & model ────────────────────────────────────────
    print("Loading data and model …")
    config, train_loader, test_loader, model = load_data_and_model(
        config_path, checkpoint_path
    )
    # calculate_accuracy(model, test_loader, device=config["device"])

    # ── 2. Extract & filter cases ───────────────────────────────────
    print("\nExtracting prediction cases …")
    explicands_info = extract_explicands_samples(
        model, test_loader, PREFIX_LEN, EXPLICANDS_NUM
    )

    # ── 3. Baselines ────────────────────────────────────────────────
    print("\nBuilding baselines …")
    avg_event, avg_sequence = build_average_event_baseline(train_loader, config)

    # ── 4. Ablation study on trace segmentation ────────────────────────────
    print("\nSegmenting traces (transition-based PELT) …")
    train_df = train_loader.dataset.log.dataframe
    unique_activities_num = train_df["activity"].nunique()
    transition_matrix = build_transition_matrix(train_df, unique_activities_num)

    activity_lookup = test_loader.dataset.log.itos["activity"]

    seg_output_dir = osp.join(OUTPUT_ROOT, "segmentation", "bpi17")

    segments_info = {}
    seg_strategies = ["transition", "random", "per_event", "two_event"]  # "transition", "random", "per_event", "two_event"
    transition_seg_kwargs = {"transition_matrix": transition_matrix}
    random_seg_kwargs = {"num_change_points": 4}

    for strategy in seg_strategies:
        print(f"\nApplying '{strategy}' segmentation strategy …")

        if strategy == "transition":
            segmentor = get_segmenter(strategy, **transition_seg_kwargs)
        elif strategy == "random":
            segmentor = get_segmenter(strategy, **random_seg_kwargs)
        else:
            segmentor = get_segmenter(strategy)

        segments_info[strategy] = {}
        palette = sns.color_palette(cc.glasbey_category10, n_colors=30)
        for name in SAMPLE_NAMES:
            
            seg_output_dir_strategy = osp.join(seg_output_dir, strategy, name)
            os.makedirs(seg_output_dir_strategy, exist_ok=True)
            
            segments_info[strategy][name] = []
            cases = explicands_info[name]["cases"]
            predictions = explicands_info[name]["predictions"]
            labels = explicands_info[name]["labels"]

            for i, (case, pred, label) in enumerate(zip(cases, predictions, labels)):
                trace = case[0, :, 0]

                segs = segmentor(trace)
                segments_info[strategy][name].append(segs)
                seg_boundaries = {
                    (seg_ind[0], seg_ind[-1]): palette[c]
                    for c, seg_ind in enumerate(segs["segment_ids"])
                }

                figure = plot_segmented_trace_vertical(
                    trace,
                    seg_boundaries,
                    activity_lookup,
                    figsize=(3, 7),
                    title=f"y_pred = {round(pred, 3)} | y_true = {label}",
                )
                figure.savefig(
                    osp.join(seg_output_dir_strategy, f"case_{i}.png"),
                    dpi=300,
                    facecolor="white",
                    edgecolor="white",
                    bbox_inches="tight",
                )
                plt.close()

    # ──  Segment-level SHAP values ───────────────────────────────
    print("\nComputing segment-level SHAP values …")
    wrapped_model = OutcomePredictorWrapper(
        model, batch_budget=1, categorical_indices=[0], device=config["device"]
    )
    fetching_fn = lambda x, hs=None: wrapped_model(sequences=x, hidden_state=hs)

    for strategy in seg_strategies:
        print(f"\nComputing SHAP values for '{strategy}' segmentation strategy …")

        sv_output_dir = osp.join(OUTPUT_ROOT, "shap_values", "bpi17", strategy)
        samples_sv_results = {}
        for name in SAMPLE_NAMES:
            
            sv_results = compute_segment_shap_values(
            fetching_fn,
            explicands_info[name]["cases"],
            segments_info[strategy][name],
            avg_event,
            config,
            output_dir=osp.join(sv_output_dir, name),
            save_plots=True)

            samples_sv_results[name] = sv_results
            print(f"  [{name.upper()}] computed SV for {len(sv_results)} cases")

        # ── 6. Persist results for ablation / correlation scripts ──────
        print("\nSaving raw results …")
        results_path = osp.join(sv_output_dir, "segment_sv_results.pkl")
        payload = {
            "sv_results": samples_sv_results,
            "segments": segments_info,
            "explicands": explicands_info,
            "baseline": avg_event,
            "config": config,
        }
        with open(results_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  Saved to {results_path}")

        print(f"\n✓ The experiment for '{strategy}' segmentation is complete.")


if __name__ == "__main__":
    main()
