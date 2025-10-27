import copy
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections.abc import Collection
from typing import Set, Callable, Union


def initial_trace_segmentation(
    case_timestamps,
    min_window_size: int = 3,
    max_window_size: int = 8,
    ts_delta_threshold: Union[float, None] = None,
):
    """
    Initial segmentation of the trace into fixed-length segments. To start with, we use even segmentation in windows of the fixed size.
    Args:
        case_timestamps : Array of shape (T, 1) containing the timestamps of events in the trace.
    """
    case_len = case_timestamps.shape[0]
    if case_timestamps.dtype != np.dtype("datetime64[ns]"):
        try:
            case_timestamps = pd.to_datetime(case_timestamps)
        except Exception as e:
            raise ValueError(
                "case_timestamps must be of datetime64[ns] dtype or convertible to datetime."
            ) from e

    ts_e0 = (
        case_timestamps.iloc[-1] - case_timestamps.iloc[0]
    ).total_seconds()  # scaled interval between the earliest and the latest events in the trace

    if ts_delta_threshold is None:
        ts_delta_threshold = max(ts_e0 / 100, 0.01)
    print(
        "The time window threshold for the initial segmentation is set to:",
        ts_delta_threshold,
    )

    segments_ids = []
    temp_segment = set()
    ts_deltas = []

    for i in range(case_len - 1, -1, -1):
        ts_i = case_timestamps.iloc[i]
        ts_j = case_timestamps.iloc[i - 1]
        ts_delta = (ts_i - ts_j).total_seconds()
        ts_deltas.append(ts_delta)

        if sum(ts_deltas) < ts_delta_threshold:
            if len(temp_segment) < max_window_size:
                temp_segment = {i}.union(temp_segment)
            else:
                segments_ids.append(temp_segment)
                temp_segment = {i}
                ts_deltas = []
        else:
            if len(temp_segment) < min_window_size:
                temp_segment = {i}.union(temp_segment)
            else:
                segments_ids.append(temp_segment)
                temp_segment = {i}
    # Add the final segment if non-empty
    if len(temp_segment) > 0:
        segments_ids.append(temp_segment)
    return sorted(segments_ids, key=min)


def split_sequence(S_init, P_prime):
    """
    Splits a segment based on a set of split points.
    Args:
        S_init: Initial sequence to be split
        P_prime: Set of split points (indices)
    """
    if not P_prime:
        return [S_init]

    # Ensure split points are sorted and add boundaries
    boundaries = sorted(set([0] + P_prime + [len(S_init)]))

    # Create subsequences
    subsequences = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        subseq = S_init[start:end]
        if len(subseq) == 0:
            print("These boundaries produced an empty subsequence:", start, end)
            raise ValueError("Empty subsequence encountered during splitting.")

        subsequences.append(subseq)

    return subsequences


def calculate_mmd_distance(seq1, seq2, sigma=1.0):
    """
    Maximum Mean Discrepancy between distributions.
    Args:
        seq1: First sequence of shape (n, feature_dim)
        seq2: Second sequence of shape (m, feature_dim)
        sigma: Bandwidth parameter for the Gaussian kernel
    """

    def gaussian_kernel(x, y, sigma=1.0):
        """Gaussian RBF kernel"""

        pairwise_sq_dists = cdist(x, y, metric="sqeuclidean")
        return np.exp(-pairwise_sq_dists / (2 * sigma**2))

    # Convert to numpy for kernel computation
    K_xx = gaussian_kernel(seq1, seq2, sigma)
    K_yy = gaussian_kernel(seq1, seq2, sigma)
    K_xy = gaussian_kernel(seq1, seq2, sigma)

    n = seq1.shape[0]
    m = seq2.shape[0]

    try:
        mmd_squared = (
            np.sum(K_xx) / (n * n) + np.sum(K_yy) / (m * m) - 2 * np.sum(K_xy) / (n * m)
        )
        return float(np.sqrt(max(mmd_squared, 0)))

    except RuntimeWarning as e:
        print(f"Division warning caught: {e}. Seq 1 shape {n}, Seq 2 shape {m}")


def extract_subsequences(trace, segments_ids):
    """
    Extract subsequences from the trace based on segment indices.
    Args:
        trace (np.ndarray): The original trace data of the shape (sequence_length, feature_dim)
        segments_ids (List[Set[int]]): List of sets containing indices for each segment.
    """
    subsequences = []
    for segment_ids in segments_ids:
        if isinstance(segment_ids, list):
            segment_ids = set().union(*segment_ids)
        ids = list(segment_ids)
        subsequences.append(trace[ids, :])
    return subsequences


def distribution_based_segmentation(
    trace: np.ndarray,
    S_init: Collection,
    K: int,
    m: int,
    D: Callable,
    embedding_generator: Callable,
) -> Set:
    """
    Distribution-based segmentation algorithm

    Args:
        S_init: Initial set/sequence
        K: Subsequence amount (target number of segments)
        D: Metric function to evaluate split quality

    Returns:
        S: Segmented subsequences
    """

    # Initialize split points
    P = set()
    if S_init is None or len(S_init) == 0:
        raise ValueError(
            "Initial sequence S_init is empty. Cannot perform segmentation."
        )
    elif len(S_init) <= K:
        raise ValueError(
            "Initial sequence S_init has fewer or equal elements than K. Cannot perform segmentation."
        )
    else:
        S = [copy.deepcopy(S_init)]  # copy.deepcopy(S_init)

    # Continue until we have K segments
    while len(S) < K:  # type: ignore
        # Initialize variables for finding best split
        distance_max = 0
        p = 0
        S_p = None

        # Try each possible split position
        for i in range(1, len(S_init) - 1):
            if i not in P:
                # Add a split point if it is not in as set of existing split points
                P_prime = sorted(P.union({i}))
                # Split the given subsequences based on the new split points
                S_prime = split_sequence(S_init, P_prime)
                # Extract subsequences from the trace according to the derived indices in S_prime
                subsequences = extract_subsequences(trace, S_prime)

                cardinality = len(subsequences)
                total_distance = 0

                # Calculate total distance between subsequences within the defined window
                for k in range(cardinality):
                    seg_k_card = len(subsequences[k])

                    start = max(k - m, 0)
                    stop = min(k + m, cardinality)

                    for j in range(start, stop):
                        seg_j_card = len(subsequences[j])
                        if k == j:
                            continue  # Skip self-comparison

                        if seg_k_card > 0 and seg_j_card > 0:
                            embedding_k = embedding_generator(subsequences[k])
                            embedding_j = embedding_generator(subsequences[j])
                            numerator = D(embedding_k, embedding_j)
                            denominator = np.sqrt(seg_k_card * seg_j_card)
                            try:
                                total_distance += numerator / denominator
                            except RuntimeWarning as e:
                                print(
                                    f"Division warning caught: {e}. Sequence segments k:{k} and j:{j} with cardinalities {seg_k_card}, {seg_j_card}"
                                )
                                # Handle the error

                if total_distance > distance_max:
                    p = i
                    distance_max = total_distance
                    S_p = S_prime
        # Update split points and segments
        P.add(p)
        # print("Added split points are ", P)
        S = S_p

    return S  # type: ignore
