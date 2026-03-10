#!/usr/bin/env python
""" """

import os
import pickle

import os.path as osp
from typing import OrderedDict
from experiments.setup_experiment import (
    load_data_and_model,
    create_loader_from_dataframe,
)

from ppm.utils import extract_explicands_samples
from local_xai.utils.trace_segmentation.transition_based import build_transition_matrix
from local_xai.utils.trace_segmentation.transition_based import (
    calculate_and_plot_segments_parallel,
)


SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")
config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth")


def save_seg_info(
    explicands_info: dict,
    output_dir: str,
    threshold: float,
    sample_name: str,
    segmentation_strategy: str,
) -> str:
    """Pickle segments_info to a file named by threshold and segmentation strategy.

    Args:
        segments_info: dict keyed by sample name (tp/tn/fp/fn) with segment results.
        output_dir: base directory (outputs/segmentation/bpi17).
        threshold: model prediction threshold used during extraction.
        segmentation_strategy: short label for the segmentation method (e.g. 'transition').

    Returns:
        Full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sample_name}_segments_thr_{threshold}_strat_{segmentation_strategy}_size_{len(explicands_info['cases'])}.pkl"
    save_path = osp.join(output_dir, filename)
    with open(save_path, "wb") as f:
        pickle.dump(explicands_info, f)
    print(f"Segments saved to {save_path}")
    return save_path



def main():
    # ── 1. Load data & model ────────────────────────────────────────
    print("Loading data and model …")
    config, train_loader, test_loader, model = load_data_and_model(
        config_path, checkpoint_path
    )
    test_df = test_loader.dataset.log.dataframe
    # calculate_accuracy(model, test_loader, device=config["device"])
    # calculate_accuracy_per_position(model, test_loader, device=config["device"])

    class_frequency = (
        test_df.groupby("case_id")["outcome"].agg(lambda x: list(x)[0]).value_counts()
    )
    class_ratio = class_frequency / class_frequency.sum()
    print("Class disribution")
    print("Accepted:", class_ratio[0].round(4), "| Cancelled", class_ratio[1].round(4))


    # ── 2. Extract & filter cases ───────────────────────────────────
    traces_len = test_df.groupby("case_id")["activity"].count()
    Q1 = traces_len.quantile(q=0.05)
    Q3 = traces_len.quantile(q=0.95)
    print(f"\nExtracting cases within 5% quantile range of their length {Q1, Q3}")
    extracted_case_ids = (
        traces_len[(traces_len > Q1) & (traces_len < Q3)].dropna().index.tolist()
    )
    extrcated_test_df = test_df[test_df["case_id"].isin(extracted_case_ids)]

    # ── 3. Calculate class distribution ───────────────────────────────────
    class_frequency = (
        extrcated_test_df.groupby("case_id")["outcome"]
        .agg(lambda x: list(x)[0])
        .value_counts()
    )
    class_ratio = class_frequency / class_frequency.sum()
    print("\nClass disribution after case length-based filtering")
    print("Accepted:", class_ratio[0].round(4), "| Cancelled", class_ratio[1].round(4))

    # TODO: encodes activity with UNK label 1 so [7, 5, 6] -> [1, 1, 1]
    # red_test_loader = deepcopy(test_loader)
    # red_test_loader.dataset.log.dataframe = extrcated_test_df.reset_index()
    original_stoi, original_itos = train_loader.dataset.log.get_vocabs()

    identity_stoi = OrderedDict(
        {feat: {v: v for v in stoi.values()} for feat, stoi in original_stoi.items()}
    )

    red_test_loader = create_loader_from_dataframe(
        extrcated_test_df, config, (identity_stoi, original_itos)
    )
    explicands_info = extract_explicands_samples(
        model,
        red_test_loader,
        prefix_len=int(Q3 + 1),
        explicands_num=None,
        threshold=0.5,
        one_offer_cases=False,
    )

    # ── 4. Transition-based segmentation ────────────────────────────
    print("\nSegmenting traces (transition-based PELT) …")
    train_df = train_loader.dataset.log.dataframe
    unique_activities_num = train_df["activity"].nunique()
    transition_matrix = build_transition_matrix(train_df, unique_activities_num)

    activity_lookup = test_loader.dataset.log.itos["activity"]

    seg_output_dir = osp.join(
        OUTPUT_ROOT, "segmentation", "bpi17", "transition_filtered_cases"
    )
    segments_info = {}
    for name in SAMPLE_NAMES:
        print(f"Starting segmentation of the sample {name}")
        os.makedirs(seg_output_dir, exist_ok=True)

        segments = calculate_and_plot_segments_parallel(
            explicands_info[name]["cases"],
            explicands_info[name]["y_pred"],
            explicands_info[name]["y_true"],
            transition_matrix,
            plot_flag=False,
            output_dir=seg_output_dir,
            activity_lookup=activity_lookup,
            n_workers=9
        )
        segments_info[name] = segments
        explicands_info[name]['segments'] = segments


        save_seg_info(
            explicands_info[name],
            output_dir=osp.join(seg_output_dir, name),
            threshold=0.7,
            sample_name=name,
            segmentation_strategy="transition",
        )


if __name__ == "__main__":
    main()
