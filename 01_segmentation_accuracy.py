import os
import ast

import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from typing import List
from collections import OrderedDict
# import seaborn as sns
# import matplotlib.pyplot as plt

from ruptures.metrics import randindex, precision_recall
from local_xai.utils.trace_segmentation.transition_based import (
    segment_trace,
    build_transition_matrix,
)

# plt.style.use("seaborn-v0_8-whitegrid")


SYNT_DATA_DIR = r"data\synthetic_data"
SYNTH_DATA_W_CP_DIR = r"outputs\synth_data_with_gt_change_points"
SEGMENTATION_DIR = r"D:\PycharmProjects\xAI-PPM\outputs\segmentation\synthetic"


def map_to_tokens(activities, mapping_dictionary):
    return [mapping_dictionary[activity] for activity in activities]


def construct_vocabulary(df, cat_columns: List = ["activity"]):
    stoi = OrderedDict()

    # Process each categorical feature
    for column in cat_columns:
        unique_values = sorted(df[column].unique().tolist())

        stoi[column] = {}

        for idx, value in enumerate(unique_values, start=0):
            stoi[column][value] = idx
    return stoi


def results_to_latex(segmentation_results: dict, out_path: str):
    """
    `segmentation_results` is expected to look like
        { sample_name: { 'randindex': [...],
                         'precision': [...],
                         'recall': [...],
                         'f1_score': [...], … },
          … }
    we summarise each list with its mean (or another aggregator)
    and write a LaTeX table.
    """
    rows = []
    for sample, stats in segmentation_results.items():
        rows.append(
            {
                "sample": sample,
                "randindex": np.mean(stats["metrics"]["randindex"]),
                "precision": np.mean(stats["metrics"]["precision"]),
                "recall": np.mean(stats["metrics"]["recall"]),
                "f1_score": np.mean(stats["metrics"]["f1_score"]),
            }
        )
    df = pd.DataFrame(rows).set_index("sample")
    # store the table
    df.to_latex(
        out_path,
        float_format="%.4f",
        caption="Segmentation experiment results",
        label="tab:segm_results",
    )
    return df


def main():
    segmentation_results = {}
    for file_name in os.listdir(SYNTH_DATA_W_CP_DIR):
        if file_name.endswith(".csv"):
            df = pd.read_csv(osp.join(SYNTH_DATA_W_CP_DIR, file_name))
            df["change_point_pos"] = df["change_point_pos"].apply(ast.literal_eval)

            sample_name = file_name.split(".")[0]

            # Build vocabulary for activities
            activity_vocab = construct_vocabulary(df)

            agg_df = df.groupby("case_nr")[["activity", "change_point_pos"]].aggregate(
                {"activity": list, "change_point_pos": lambda x: x.iloc[0] + [len(x)]}
            )
            agg_df["tokens"] = agg_df["activity"].apply(
                lambda x: map_to_tokens(x, activity_vocab["activity"])
            )

            gt_brkps = agg_df["change_point_pos"]

            # Build transition matrix
            transition_matrix = build_transition_matrix(
                agg_df,
                unique_activities_num=len(activity_vocab["activity"]),
                special_tokens_num=3,
                case_id_col="case_nr",
                activity_col="tokens",
            )

            seg_result = []
            metrics = {"randindex": [], "precision": [], "recall": [], "f1_score": []}

            print(
                "Evaluating segmentation quality of the sample {}...".format(
                    sample_name
                )
            )
            for trace, gt_brkps_i in tqdm(
                zip(agg_df["tokens"][:1000], gt_brkps[:1000])
            ):
                result = segment_trace(np.asarray(trace), transition_matrix)
                seg_result.append(result)

                # Calculating randindex and precision-recall for each trace
                randind_i = randindex(result, gt_brkps_i)
                pr_i = precision_recall(gt_brkps_i, result, margin=1)
                f1_i = (
                    (2 * pr_i[0] * pr_i[1]) / (pr_i[0] + pr_i[1])
                    if (pr_i[0] + pr_i[1]) > 0
                    else 0
                )

                metrics["randindex"].append(randind_i)
                metrics["precision"].append(pr_i[0])
                metrics["recall"].append(pr_i[1])
                metrics["f1_score"].append(f1_i)

            print(
                "Average RankIndex of the sample {}:".format(sample_name),
                np.mean(metrics["randindex"]).round(4),
            )
            print(
                "Average F1-score of the sample {}:".format(sample_name),
                np.mean(metrics["f1_score"]).round(4),
            )

            segmentation_results[sample_name] = {
                "breakpoints": seg_result,
                "gt_breakpoints": gt_brkps.tolist(),
                "metrics": metrics,
            }

    results_to_latex(
        segmentation_results, osp.join(SEGMENTATION_DIR, "segm_quality_results.tex")
    )
    # print("wrote", latex_df.shape, "table to outputs/segm_results.tex")


if __name__ == "__main__":
    data_file = "dataloan_log_['choose_procedure']_100000_train_normal.csv"
    main()
