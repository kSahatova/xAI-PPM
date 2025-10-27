import numpy as np
from typing import List


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
    current_segment = [trace[0]]

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
        else:
            # Low probability transition - start new segment
            segments.append(current_segment)
            current_segment = [trace[i + 1]]

    # Add final segment
    segments.append(current_segment)
    return segments


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
