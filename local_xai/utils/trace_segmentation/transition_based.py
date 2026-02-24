import numpy as np
import pandas as pd
import os.path as osp
import ruptures as rpt
from ruptures.base import BaseCost
import seaborn as sns
import matplotlib.pyplot as plt
from local_xai.utils.trace_segmentation.visualization import plot_segmented_trace_vertical


def build_transition_matrix(
    log_df: pd.DataFrame, unique_activities_num: int = 25, special_tokens_num: int = 3
):
    n_cols = n_rows = special_tokens_num + unique_activities_num
    adjacency = np.zeros((n_rows, n_cols), dtype=int)

    for case_id, group in log_df.groupby("case_id"):
        activities = group["activity"].values
        for i in range(len(activities) - 1):
            adjacency[activities[i], activities[i + 1]] += 1

    transition_matrix = adjacency / adjacency.sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix).astype(np.float64)

    return transition_matrix


class TransitionCost(BaseCost):
    """
    Custom cost function for ruptures based on transition probabilities.
    """

    # avoid overriding BaseCost.model property - use a different attribute name
    model = "custom"

    def __init__(self, trans_matrix, min_size: int = 2):
        self.trans_matrix = trans_matrix
        self.min_size = min_size

    def fit(self, signal):
        """signal should be the trace"""
        self.signal = signal

        return self

    def error(self, start, end):
        """
        Compute the cost (negative log-likelihood) of segment [start:end].
        Lower cost = better fit.
        """
        segment = self.signal[start:end]

        if len(segment) < 2:
            return 0

        # Compute negative log-likelihood
        cost = 0
        for i in range(len(segment) - 1):
            from_activity = int(segment[i])
            to_activity = int(segment[i + 1])
            prob = self.trans_matrix[from_activity, to_activity]

            if prob > 0:
                cost -= np.log(prob)
            else:
                print(
                    "Probaility  of the transition is zero:",
                    from_activity,
                    "->",
                    to_activity,
                )
                cost += (
                    10  # Penalty for unseen transitions # TODO: check this penalty!!
                )
        return cost


def segment_trace(
    trace, transition_matrix, penalty_params={"start": 0, "stop": 1, "step": 0.01}
):
    optimizer = rpt.Pelt(
        custom_cost=TransitionCost(transition_matrix), jump=1, min_size=1
    )
    optimizer.fit(trace)

    penalties = np.arange(
        penalty_params["start"], penalty_params["stop"], penalty_params["step"]
    )
    results = [optimizer._seg(penalty) for penalty in penalties]
    cost_list = [sum(result.values()) for result in results]
    seg_num_list = [len(result) for result in results]

    best_aic = 1e10
    best_penalty = 0

    for i, (cost, segments_num) in enumerate(zip(cost_list, seg_num_list)):
        aic = 2 * segments_num - 2 * np.log(cost)
        if aic < best_aic:
            best_aic = aic
            best_penalty = penalties[i]

    breakpoints = optimizer.fit(trace).predict(pen=best_penalty)

    return breakpoints


def calculate_and_plot_segments(
    cases,
    predictions,
    labels,
    transition_matrix: np.ndarray,
    plot_flag: bool = True,
    output_dir: str = "",
    activity_lookup: dict = {},
    palette: sns.palettes._ColorPalette = sns.color_palette(),
):
    segments = []

    for i, (case, pred, label) in enumerate(zip(cases, predictions, labels)):
        trace = case[0, :, 0]
        breakpoints = segment_trace(trace, transition_matrix)

        trace_segments = []
        trace_seg_ids = []
        for j, t in enumerate(trace):
            if j == 0 or j in breakpoints:
                trace_segments.append([t])
                trace_seg_ids.append([j])
            else:
                trace_segments[-1].append(t)
                trace_seg_ids[-1].append(j)
        segments.append({"segments": trace_segments, "segment_ids": trace_seg_ids})

        if plot_flag:
            seg_boundaries = {}
            for c, seg in enumerate(trace_seg_ids):
                seg_boundaries[(seg[0], seg[-1])] = palette[c]
            figure = plot_segmented_trace_vertical(
                trace,
                figsize=(3, 7),
                seg_boundaries_colors=seg_boundaries,
                activity_lookup=activity_lookup,
                title=f"y_pred = {round(pred, 3)} | y_true = {label}",
            )
            figure.savefig(
                osp.join(output_dir, f"case_{i}.png"),
                dpi=300,
                facecolor="white",
                edgecolor="white",
                bbox_inches="tight",
            )
            plt.close()

    return segments
