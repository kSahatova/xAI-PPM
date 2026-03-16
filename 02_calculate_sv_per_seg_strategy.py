#!/usr/bin/env python
""" """

import os
import pickle
import pandas as pd
import os.path as osp
from typing import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
from experiments.setup_experiment import (
    load_data_and_model,
    create_loader_from_dataframe,
)

from ppm.utils import (
    calculate_accuracy_per_position,
    extract_explicands_samples,
    calculate_accuracy,
    calculate_auc
)

from local_xai.utils.trace_segmentation.segmentation_factory import (
    apply_segmenter_parallel,
)
from local_xai.utils.trace_segmentation.transition_based import build_transition_matrix
from local_xai.seqshap.calulate_seqshap_parallel import (
    compute_segment_shap_values_parallel,
)
from local_xai.utils.baseline_calculation import build_average_event_baseline
from local_xai.utils.trace_segmentation.visualization import plot_transition_matrix
from timeshap.wrappers.outcome_predictor_wrapper import OutcomePredictorWrapper


SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]

SEG_STRATEGIES = ["random"]
# SEG_STRATEGIES = ["per_event", "distribution", "transition"]
PREFIX_LEN = 30
# PREFIX_LEN = 15
MODEL_CONFIDENCE_THRESHOLD = 0.7

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")
config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
# config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op_sepsis.txt")
checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth")
# checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\rnn_outcome_sepsis.pth")


def main():
    # ── 1. Load data & model ────────────────────────────────────────
    print("=" * 80)
    print("Loading data and model")
    print("=" * 80)
    config, train_loader, test_loader, model = load_data_and_model(
        config_path, checkpoint_path
    )
    calculate_accuracy(model, test_loader, device=config["device"])
    calculate_accuracy_per_position(model, test_loader, device=config["device"], 
                                    save_path=r'D:\PycharmProjects\xAI-PPM\accuracy_per_position_sepsis.png')
    calculate_auc(model, test_loader, config["device"])

    ds_name = config["dataset"].lower()

    train_df = train_loader.dataset.log.dataframe
    unique_activities_num = train_df["activity"].nunique()
    transition_matrix = build_transition_matrix(train_df, unique_activities_num)
    # plot_transition_matrix(transition_matrix)

    avg_event, _ = build_average_event_baseline(train_loader, config)
    wrapped_model = OutcomePredictorWrapper(
        model, batch_budget=1, categorical_indices=[0], device=config["device"]
    )
    fetching_fn = lambda x, hs=None: wrapped_model(sequences=x, hidden_state=hs)

    # ── 2. Extract & filter cases ───────────────────────────────────
    test_df = test_loader.dataset.log.dataframe
    traces_len = test_df.groupby("case_id")["activity"].count()
    Q1 = traces_len.quantile(q=0.05)
    Q3 = traces_len.quantile(q=0.95)

    original_stoi, original_itos = train_loader.dataset.log.get_vocabs()
    identity_stoi = OrderedDict(
        {feat: {v: v for v in stoi.values()} for feat, stoi in original_stoi.items()}
    )

    # ── 3c. Extract explicands with the defined prefix length ─────
    print("=" * 80)
    print("Extracting cases")
    print("=" * 80)
    case_ids = traces_len[(traces_len > Q1) & (traces_len < Q3)].dropna().index.tolist()
    red_test_df = test_df[test_df["case_id"].isin(case_ids)]

    print(f"Number of extracted cases: {len(case_ids)}")
    class_frequency = (
        red_test_df.groupby("case_id")["outcome"]
        .agg(lambda x: list(x)[0])
        .value_counts()
    )
    class_ratio = class_frequency / class_frequency.sum()
    print("\nClass disribution after the filtering of outliers")
    if ds_name == "bpi17":
        print(
            "Accepted:", class_ratio[0].round(4), "| Cancelled", class_ratio[1].round(4)
        )
    elif ds_name == "sepsis":
        print(
            "No Return ER",
            class_ratio[0].round(4),
            "| Return ER:",
            class_ratio[1].round(4),
        )

    red_test_loader = create_loader_from_dataframe(
        red_test_df, config, (identity_stoi, original_itos)
    )
    explicands_info = extract_explicands_samples(
        model,
        red_test_loader,
        prefix_len=PREFIX_LEN,
        explicands_num=None,
        threshold=MODEL_CONFIDENCE_THRESHOLD,
        one_offer_cases=False,
    )

    # ── 4. Segment extracted cases ─────
    print("=" * 80)
    print("Starting trace segmentation")
    print("=" * 80)
    seg_strategy_kwargs = {
        "transition": {"transition_matrix": transition_matrix},
        "distribution": {
            "min_window_size": 3,
            "max_window_size": 3,
            "m": 5,  # measuring window,
            "timestamp": test_loader.dataset.timestamps,
        },
        "per_event": {},
        "random": {"num_change_points": 8, "seed": 32},
    }

    for seg_strategy in SEG_STRATEGIES:
        random_seed =  seg_strategy_kwargs['random']['seed']
        sv_output_dir = osp.join(
            OUTPUT_ROOT, "shap_values", ds_name, f"{seg_strategy}_cohort_medium_seed{random_seed}"
        )

        for name in SAMPLE_NAMES:
            print(
                f"Applying '{seg_strategy}' segmentation strategy to the sample {name}"
            )

            segments = apply_segmenter_parallel(
                seg_strategy,
                explicands_info[name]["cases"],
                n_workers=9,
                **seg_strategy_kwargs.get(seg_strategy, {}),
            )
            explicands_info[name]["segments"] = segments

            sv_results = compute_segment_shap_values_parallel(
                fetching_fn,
                explicands_info[name]["cases"],
                explicands_info[name]["segments"],
                avg_event,
                config,
                output_dir=sv_output_dir,
                save_plots=False,
                n_workers=9,
            )
            explicands_info[name]["sv"] = sv_results

            os.makedirs(sv_output_dir, exist_ok=True)
            with open(
                osp.join(sv_output_dir, f"{name}_segment_sv_results.pkl"), "wb"
            ) as f:
                pickle.dump(explicands_info[name], f)
            print(f"  Saved to {sv_output_dir}")


if __name__ == "__main__":
    main()
