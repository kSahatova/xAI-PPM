#!/usr/bin/env python
""" """

import os
import pickle
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from typing import Dict, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
from experiments.setup_experiment import (
    load_data_and_model,
    create_loader_from_dataframe,
)

from ppm.utils import extract_explicands_samples
from local_xai.utils.trace_segmentation.transition_based import build_transition_matrix
from local_xai.utils.trace_segmentation.transition_based import (
    calculate_and_plot_segments_parallel,
)
from local_xai.seqshap.calulate_seqshap_parallel import (
    compute_segment_shap_values_parallel,
)
from local_xai.utils.baseline_calculation import build_average_event_baseline
from timeshap.wrappers.outcome_predictor_wrapper import OutcomePredictorWrapper


SAMPLE_NAMES = ["tp", "fp", "fn", "tn"]
PREFIX_LEN = 15
EXPLICANDS_NUM = 10

PROJECT_DIR = r"D:\PycharmProjects\xAI-PPM"
OUTPUT_ROOT = osp.join(PROJECT_DIR, r"outputs")
config_path = osp.join(PROJECT_DIR, r"configs\explain_lstm_args_for_op.txt")
checkpoint_path = osp.join(OUTPUT_ROOT, r"checkpoints\BPI17_rnn_outcome_bpi17.pth")


def build_cohort_length_df(
    test_df: pd.DataFrame,
    Q1: float,
    Q3: float,
) -> pd.DataFrame:
    """Build a flat DataFrame with trace_length, cohort, and actual_outcome per case.

    Cohorts:
        'short'         – traces shorter than Q1
        'medium'   – traces between Q1 and Q3
        'long'         – traces longer than Q3

    actual_outcome is derived from the 'outcome' column (0 = accepted, 1 = cancelled).
    """
    traces_len = test_df.groupby("case_id")["activity"].count()
    outcome_per_case = test_df.groupby("case_id")["outcome"].agg(lambda x: list(x)[0])

    conditions = {
        "short": traces_len < Q1,
        "medium": (traces_len >= Q1) & (traces_len <= Q3),
        "long": traces_len > Q3,
    }

    rows = []
    for cohort_label, mask in conditions.items():
        for case_id in traces_len[mask].index:
            outcome_code = outcome_per_case[case_id]
            actual_outcome = "cancelled" if outcome_code == 1 else "accepted"
            rows.append(
                {
                    "case_id": case_id,
                    "trace_length": int(traces_len[case_id]),
                    "cohort": cohort_label,
                    "actual_outcome": actual_outcome,
                }
            )
    return pd.DataFrame(rows)


def plot_cohort_length_boxplots(
    test_df: pd.DataFrame,
    Q1: float,
    Q3: float,
    output_path: str = "",
) -> plt.Figure:
    """Boxplots of case lengths per cohort, split by accepted / cancelled.

    Layout: one panel per cohort ("<Q1", ">Q1 & <Q3", ">Q3").
    Within each panel, two boxes – one per actual outcome class.
    """
    df = build_cohort_length_df(test_df, Q1, Q3)

    cohort_order = ["short", "medium", "long"]
    outcome_order = ["accepted", "cancelled"]
    palette = {"accepted": "#3E8540B6", "cancelled": "#3D7AACB6"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)

    for ax, cohort in zip(axes, cohort_order):
        sub = df[df["cohort"] == cohort]

        sns.boxplot(
            data=sub,
            x="actual_outcome",
            y="trace_length",
            order=outcome_order,
            palette=palette,
            width=0.5,
            linewidth=1.4,
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="actual_outcome",
            y="trace_length",
            order=outcome_order,
            palette=palette,
            size=3,
            alpha=0.35,
            jitter=True,
            dodge=False,
            ax=ax,
        )

        # annotate n per box
        for i, outcome in enumerate(outcome_order):
            grp = sub[sub["actual_outcome"] == outcome]["trace_length"]
            if grp.empty:
                continue
            ax.text(
                i + 0.28,
                grp.median(),
                f"n={len(grp)}\nmedian={grp.median():.0f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        ax.set_title(f"Cohort: {cohort} cases", fontsize=12)
        ax.set_xlabel("Actual outcome", fontsize=11)
        ax.set_ylabel("Case length" if ax is axes[0] else "", fontsize=11)
        sns.despine(ax=ax)

    fig.suptitle(
        "Case length distributions by cohort and actual outcome class",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")

    return fig


def plot_trace_length_distribution(
    traces_len: pd.Series,
    Q1: float,
    Q3: float,
    output_path: str = "",
) -> plt.Figure:
    """Histogram + KDE of trace lengths with Q1/Q3 boundary lines marked."""
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(traces_len, bins=40, kde=True, color="#4C72B0", alpha=0.55, ax=ax)

    ax.axvline(Q1, color="#E04B3A", linestyle="--", linewidth=1.5, label=f"Q5  = {Q1:.0f}")
    ax.axvline(Q3, color="#2CA02C", linestyle="--", linewidth=1.5, label=f"Q95 = {Q3:.0f}")
    ax.axvline(traces_len.median(), color="orange", linestyle=":", linewidth=1.5,
               label=f"median = {traces_len.median():.0f}")

    ax.set_xlabel("Trace length (# events)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of trace lengths in the test set", fontsize=13)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")

    return fig


def main():
    # ── 1. Load data & model ────────────────────────────────────────
    print("Loading data and model …")
    config, train_loader, test_loader, model = load_data_and_model(
        config_path, checkpoint_path
    )

    # ── 2. Transition-based segmentation ────────────────────────────
    print("\nSegmenting traces (transition-based PELT) …")
    train_df = train_loader.dataset.log.dataframe
    unique_activities_num = train_df["activity"].nunique()
    transition_matrix = build_transition_matrix(train_df, unique_activities_num)

    activity_lookup = test_loader.dataset.log.itos["activity"]

    print("\nBuilding a baseline …")
    avg_event, avg_sequence = build_average_event_baseline(train_loader, config)

    print("\nComputing segment-level SHAP values …")
    wrapped_model = OutcomePredictorWrapper(
        model, batch_budget=1, categorical_indices=[0], device=config["device"]
    )
    fetching_fn = lambda x, hs=None: wrapped_model(sequences=x, hidden_state=hs)

    # ── 3. Extract & filter cases ───────────────────────────────────
    test_df = test_loader.dataset.log.dataframe
    traces_len = test_df.groupby("case_id")["activity"].count()
    Q1 = traces_len.quantile(q=0.05)
    Q3 = traces_len.quantile(q=0.95)

    # ── 3a. Trace-length distribution ───────────────────────────────
    plot_trace_length_distribution(
        traces_len,
        Q1,
        Q3,
        output_path=osp.join(OUTPUT_ROOT, "segmentation", "bpi17", "trace_length_distribution.png"),
    )
    plt.show()

    # ── 3b. Boxplots of case lengths per cohort × outcome class ─────
    plot_cohort_length_boxplots(
        test_df,
        Q1,
        Q3,
        output_path=osp.join(OUTPUT_ROOT, "segmentation", "bpi17", "cohort_length_boxplots.png"),
    )
    plt.show()

    original_stoi, original_itos = train_loader.dataset.log.get_vocabs()

    identity_stoi = OrderedDict(
        {feat: {v: v for v in stoi.values()} for feat, stoi in original_stoi.items()}
    )

    # ── 3c. Extract explicands from each cohort with the defined prefix length ─────
    cohorts_info = {}
    conditions = {
        # "short": [(traces_len < Q1), 15],
        "medium": [((traces_len > Q1) & (traces_len < Q3)), 30],
        # "long": [(traces_len > Q3), 70],
    }

    for cohort, (condition, prefix_length) in conditions.items():
        print(f"\n**********Extracting {cohort} cases**********")
        cohort_case_ids = traces_len[condition].dropna().index.tolist()
        cohort_test_df = test_df[test_df["case_id"].isin(cohort_case_ids)]
        cohorts_info[cohort] = cohort_test_df
        print(f"Number of cases in the cohort '{cohort}' cases: ", len(cohort_case_ids))
        class_frequency = (
            cohort_test_df.groupby("case_id")["outcome"]
            .agg(lambda x: list(x)[0])
            .value_counts()
        )
        class_ratio = class_frequency / class_frequency.sum()
        print("\nClass disribution after case length-based filtering")
        print(
            "Accepted:", class_ratio[0].round(4), "| Cancelled", class_ratio[1].round(4)
        )

        red_test_loader = create_loader_from_dataframe(
            cohort_test_df, config, (identity_stoi, original_itos)
        )
        explicands_info = extract_explicands_samples(
            model,
            red_test_loader,
            prefix_len=prefix_length,
            explicands_num=None,
            threshold=0.7,
            one_offer_cases=False,
        )

        # ── 4. Segment extracted cases ─────

        seg_output_dir = osp.join(
            OUTPUT_ROOT, "segmentation", "bpi17", f"transition_cohort_{cohort}"
        )
        sv_output_dir = osp.join(
            OUTPUT_ROOT, "shap_values", "bpi17", f"transition_cohort_{cohort}"
        )

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
                n_workers=9,
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
                n_workers=9
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
