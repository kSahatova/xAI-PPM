import numpy as np
from typing import List
from ruptures.base import BaseCost
import ruptures as rpt


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
    trace: np.ndarray, cost_object: BaseCost, jump: int = 1, min_size: int = 1
):
    """
    Segments the input trace with the transition based PELT algorithm
    """
    optimizer = rpt.Pelt(custom_cost=cost_object, jump=jump, min_size=min_size)
    optimizer.fit(trace)

    penalties = np.arange(0, 1, 0.01)
    results = [optimizer._seg(penalty) for penalty in penalties]
    cost_list = [sum(result.values()) for result in results]
    seg_num_list = [len(result) for result in results]

    best_aic = 1e10
    best_penalty = 0
    best_segmentation = {}

    for i, (cost, segments_num) in enumerate(zip(cost_list, seg_num_list)):
        aic = 2 * segments_num - 2 * np.log(cost)
        if aic < best_aic:
            best_aic = aic
            best_penalty = penalties[i].round(3)
            best_segmentation = results[i]

    breakpoints = [tup[1] for tup in list(best_segmentation.keys())]
    return {"breakpoints": breakpoints, "penalty": best_penalty}


def segment_by_threshold(
    trace: np.ndarray, transition_matrix: np.ndarray, threshold: float = 0.5
):  # -> List[List[int]]
    """
    Segment when transition probability drops below threshold.

    Strategy: Break segment when P(event_i+1 | event_i) < threshold
    High threshold = more segments (stricter about "high probability")
    Low threshold = fewer segments (more permissive)
    """

    segments = []
    segments_ids = []
    current_segment = [trace[0]]
    current_segment_ids = {0}

    for i in range(len(trace) - 1):
        activity1 = trace[i]
        activity2 = trace[i + 1]

        if isinstance(activity1, float):
            activity1 = int(activity1)

        if isinstance(activity2, float):
            activity2 = int(activity2)

        prob = transition_matrix[activity1, activity2]

        if prob >= threshold:
            # High probability transition - continue segment
            current_segment.append(trace[i + 1])
            current_segment_ids.union({i + 1})
        else:
            # Low probability transition - start new segment
            segments.append(current_segment)
            segments_ids.append(current_segment_ids)

            current_segment = [trace[i + 1]]
            current_segment_ids = {i + 1}

    # Add final segment
    segments.append(current_segment)
    segments_ids.append(current_segment_ids)
    return segments, segments_ids


def segment_by_adaptive_threshold(
    trace: List[str], transition_matrix: np.ndarray, percentile: float = 25
) -> List[List[str]]:
    """
    Segment using adaptive threshold based on trace-specific probabilities.

    Strategy: Calculate all transition probabilities in the trace,
    then break at transitions below the Nth percentile.

    This adapts to each trace's probability distribution.
    """
    if len(trace) <= 1:
        return [trace]

    # Get all transition probabilities for this trace
    probs = [
        transition_matrix[int(trace[i]), int(trace[i + 1])]
        for i in range(len(trace) - 1)
    ]

    # Calculate adaptive threshold
    threshold = np.percentile(probs, percentile)

    segments = []
    current_segment = [trace[0]]

    for i in range(len(trace) - 1):
        if probs[i] >= threshold:
            current_segment.append(trace[i + 1])
        else:
            segments.append(current_segment)
            current_segment = [trace[i + 1]]

    segments.append(current_segment)
    return segments
