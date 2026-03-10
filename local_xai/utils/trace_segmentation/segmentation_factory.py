"""
Segmentation factory for trace prefixes.

Provides a unified interface for all segmentation strategies via
``get_segmenter(mode, **kwargs)`` which returns a callable:

    segmenter(trace) -> {"segments": [...], "segment_ids": [...]}

Supported modes:
  - "transition" : control-flow-aware PELT segmentation
  - "random"     : random change points
  - "per_event"  : each event is its own segment  (length 1)
  - "two_event"  : consecutive pairs of events    (length 2)

Usage:
    >>> from segmentation_factory import get_segmenter
    >>>
    >>> # Transition-based (requires a pre-built transition matrix)
    >>> seg = get_segmenter("transition", transition_matrix=T)
    >>> seg(trace)
    >>>
    >>> # Random with fixed number of change points
    >>> seg = get_segmenter("random", num_change_points=3, seed=42)
    >>> seg(trace)
    >>>
    >>> # Per-event
    >>> seg = get_segmenter("per_event")
    >>> seg(trace)
    >>>
    >>> # Iterate over all strategies in an ablation study
    >>> for mode in get_available_modes():
    ...     seg = get_segmenter(mode, transition_matrix=T, seed=42)
    ...     result = seg(trace)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .transition_based import segment_trace


# ── Public types ────────────────────────────────────────────────────
SegmentResult = Dict[str, List[List]]
Segmenter = Callable[[np.ndarray], SegmentResult]

_MODES = ("transition", "distribution", "random", "per_event", "two_event")


# ── Factory ─────────────────────────────────────────────────────────

def get_available_modes() -> tuple[str, ...]:
    """Return the names of all supported segmentation modes."""
    return _MODES


def get_segmenter(mode: str, **kwargs) -> Segmenter:
    """Create and return a segmenter function for the given mode.

    Parameters
    ----------
    mode : str
        One of "transition", "distribution", "random", "per_event", "two_event".

    Keyword arguments (mode-dependent):
        transition_matrix : np.ndarray
            Required for "transition" mode.
        K : int | None
            Target number of segments for "distribution" mode. If None (default),
            computed per trace as ``int(min(max(10, n/2), 30))``.
        m : int
            Comparison window half-width for "distribution" mode (default 2).
        sigma : float
            MMD Gaussian kernel bandwidth for "distribution" mode (default 1.0).
        num_change_points : int | None
            Optional for "random" mode. Fixed number of change points.
        min_change_points : int
            Optional for "random" mode (default 1).
        seed : int | None
            Optional for "random" mode.

    Returns
    -------
    Segmenter
        A callable ``(trace) -> {"segments": [...], "segment_ids": [...]}``
    """
    if mode == "transition":
        return _make_transition_segmenter(**kwargs)
    elif mode == "distribution":
        return _make_distribution_segmenter(**kwargs)
    elif mode == "random":
        return _make_random_segmenter(**kwargs)
    elif mode == "per_event":
        return _make_per_event_segmenter()
    elif mode == "two_event":
        return _make_two_event_segmenter()
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Available: {_MODES}"
        )


# ── Transition-based ────────────────────────────────────────────────

def _make_transition_segmenter(
    transition_matrix: np.ndarray = None, **_extra
) -> Segmenter:
    if transition_matrix is None:
        raise ValueError(
            "transition_matrix is required for mode='transition'"
        )
    tm = transition_matrix

    def _segment(trace: np.ndarray) -> SegmentResult:
        seq = _to_list(trace)
        n = len(seq)
        if n == 0:
            return {"segments": [], "segment_ids": []}

        breakpoints = segment_trace(np.asarray(seq), tm)

        segments, segment_ids = [], []
        for j, val in enumerate(seq):
            if j == 0 or j in breakpoints:
                segments.append([float(val)])
                segment_ids.append([j])
            else:
                segments[-1].append(float(val))
                segment_ids[-1].append(j)

        return {"segments": segments, "segment_ids": segment_ids}

    return _segment


# ── Distribution-based ──────────────────────────────────────────────

def _make_distribution_segmenter(
    K: Optional[int] = None,
    m: int = 2,
    sigma: float = 1.0,
    timestamps: Optional[np.ndarray] = None,
    min_window_size: int = 3,
    max_window_size: int = 8,
    ts_delta_threshold: Optional[float] = None,
    **_extra,
) -> Segmenter:
    """Distribution-based segmentation using MMD as the distance measure.

    Parameters
    ----------
    K : int | None
        Target number of segments. If None (default), computed per trace as
        ``int(min(max(10, n / 2), 30))`` where n is the trace length.
    m : int
        Half-width of the comparison window (how many neighbouring segments
        are compared when evaluating a split candidate).
    sigma : float
        Bandwidth of the Gaussian kernel used in MMD.
    timestamps : np.ndarray | None
        Optional datetime array of shape (n,) used for the initial
        time-aware segmentation step. When provided, ``initial_trace_segmentation``
        is called to produce ``S_init`` before the distribution-based pass.
        When None (default), a uniform ``list(range(n))`` is used.
    min_window_size : int
        Passed to ``initial_trace_segmentation`` (default 3).
    max_window_size : int
        Passed to ``initial_trace_segmentation`` (default 8).
    ts_delta_threshold : float | None
        Passed to ``initial_trace_segmentation``. When None, the function
        derives the threshold automatically from the trace span.
    """
    from .distribution_based import (
        distribution_based_segmentation,
        calculate_mmd_distance,
    )

    def _segment(trace: np.ndarray) -> SegmentResult:
        seq = _to_list(trace)
        n = len(seq)
        if n == 0:
            return {"segments": [], "segment_ids": []}

        k_target = K if K is not None else int(min(max(10, n / 2), 30))
        k = min(k_target, n - 1)
        if k < 2:
            return {"segments": [[float(v) for v in seq]], "segment_ids": [list(range(n))]}

        # ── Initial segmentation ────────────────────────────────────────
        if timestamps is not None:
            from .distribution_based import initial_trace_segmentation
            ts_series = pd.Series(timestamps)
            initial_segs = initial_trace_segmentation(
                ts_series,
                min_window_size=min_window_size,
                max_window_size=max_window_size,
                ts_delta_threshold=ts_delta_threshold,
            )
            # Flatten the list of sets into an ordered list of index sets for S_init
            S_init = [sorted(seg) for seg in initial_segs]
            # Re-derive K from the initial segmentation if not provided
            if K is None:
                k_target = int(min(max(10, len(initial_segs) / 2), 30))
                k = min(k_target, n - 1)
        else:
            # S_init: each timestep as a singleton set so extract_subsequences
            # can union them via set().union(*segment_ids)
            S_init = [{i} for i in range(n)]

        # Reshape 1-D activity trace to (n, 1) so extract_subsequences returns 2-D arrays
        trace_2d = np.array(seq, dtype=float).reshape(-1, 1)

        S = distribution_based_segmentation(
            trace=trace_2d,
            S_init=S_init,
            K=k,
            m=m,
            D=lambda x, y: calculate_mmd_distance(x, y, sigma=sigma),
            embedding_generator=lambda subseq: subseq,  # already 2-D from extract_subsequences
        )

        # S is a list of sublists; each sublist holds the S_init elements assigned to that
        # segment (sets or sorted lists). Flatten each into a sorted list of ints.
        segment_ids = [
            sorted(set().union(*[s if isinstance(s, set) else set(s) for s in seg]))
            for seg in S
        ]
        segments = [[float(trace_2d[idx, 0]) for idx in ids] for ids in segment_ids]
        return {"segments": segments, "segment_ids": segment_ids}

    return _segment



# ── Random ──────────────────────────────────────────────────────────

def _make_random_segmenter(
    num_change_points: Optional[int] = None,
    min_change_points: int = 1,
    seed: Optional[int] = None,
    **_extra,
) -> Segmenter:
    ncp = num_change_points
    mcp = min_change_points
    s = seed

    def _segment(trace: np.ndarray) -> SegmentResult:
        seq = _to_list(trace)
        n = len(seq)
        if n == 0:
            return {"segments": [], "segment_ids": []}

        rng = np.random.default_rng(s)
        max_cp = max(0, n - 1)

        if ncp is None:
            lo, hi = max(1, mcp), max_cp
            if lo > hi:
                cps = []
            else:
                k = int(rng.integers(lo, hi + 1))
                cps = sorted(
                    rng.choice(np.arange(1, n), size=k, replace=False).tolist()
                )
        else:
            k = int(max(0, min(ncp, max_cp)))
            cps = (
                sorted(rng.choice(np.arange(1, n), size=k, replace=False).tolist())
                if k else []
            )

        return _boundaries_to_result(seq, cps)

    return _segment


# ── Per-event ───────────────────────────────────────────────────────

def _make_per_event_segmenter() -> Segmenter:

    def _segment(trace: np.ndarray) -> SegmentResult:
        seq = _to_list(trace)
        segments = [[float(v)] for v in seq]
        segment_ids = [[i] for i in range(len(seq))]
        return {"segments": segments, "segment_ids": segment_ids}

    return _segment


# ── Two-event ──────────────────────────────────────────────────────

def _make_two_event_segmenter():

    def _segment(trace: np.ndarray) -> dict:
        seq = _to_list(trace)
        n = len(seq)
        if n == 0:
            return {"segments": [], "segment_ids": []}

        segments, segment_ids = [], []

        for i in range(0, n, 2):
            end = min(i + 2, n)
            ids = [i, end]
            vals = [float(seq[j]) for j in range(i, end)]
            segment_ids.append(ids)
            segments.append(vals)

        # If last segment has only one element, merge it into the previous one
        if len(segments) > 1 and len(segments[-1]) == 1:
            segments[-2].extend(segments[-1])
            segment_ids[-2][-1] = segment_ids[-1][-1]
            segments.pop()
            segment_ids.pop()

        return {"segments": segments, "segment_ids": segment_ids}

    return _segment

# ── Shared helpers ──────────────────────────────────────────────────

def _to_list(trace) -> list:
    if isinstance(trace, (pd.Series, np.ndarray)):
        return trace.tolist()
    return list(trace)


def _boundaries_to_result(seq: list, cps: list) -> SegmentResult:
    """Convert a sorted list of change-point indices into segments."""
    boundaries = []
    start = 0
    for cp in cps:
        # if cp - start == 1:
        #     boundaries.append((start, cp))
        # else:
        boundaries.append((start, cp - 1))
        start = cp
    boundaries.append((start, len(seq) - 1))

    segments, segment_ids = [], []
    for s, e in boundaries:
        ids = list(range(s, e + 1))
        vals = [float(seq[i]) for i in ids]
        segment_ids.append(ids)
        segments.append(vals)

    return {"segments": segments, "segment_ids": segment_ids}


# ── Parallel batch application ──────────────────────────────────────

def _parallel_worker(args):
    """Top-level worker so it can be pickled by ProcessPoolExecutor."""
    mode, kwargs, i, case = args
    segmenter = get_segmenter(mode, **kwargs)
    trace = case[0, :, 0]
    return i, segmenter(trace)


def apply_segmenter_parallel(mode: str, cases, n_workers=None, **kwargs) -> list:
    """Apply any segmentation strategy in parallel across cases.

    Parameters
    ----------
    mode : str
        Segmentation mode passed to get_segmenter.
    cases : sequence
        Each element is a case array with shape (1, seq_len, n_features).
    n_workers : int | None
        Number of worker processes. Defaults to os.cpu_count().
    **kwargs
        Forwarded to get_segmenter (e.g. transition_matrix=..., seed=...).

    Returns
    -------
    list of SegmentResult
        Order matches the input cases.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    args_list = [(mode, kwargs, i, case) for i, case in enumerate(cases)]
    results: List[Optional[SegmentResult]] = [None] * len(args_list)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_parallel_worker, a): a[2] for a in args_list}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{mode} segmentation (parallel)",
        ):
            i, segment = future.result()
            results[i] = segment

    return results


# ── Self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dummy transition matrix for testing (not meaningful, just right shape)
    n_act = 30
    dummy_tm = np.random.rand(n_act, n_act)
    dummy_tm = dummy_tm / dummy_tm.sum(axis=1, keepdims=True)

    trace = np.array([7, 11, 24, 24, 24, 24, 24, 23, 23, 6, 23, 23, 23, 3, 15])

    for mode in get_available_modes()[1:]:  # Skip "transition" for this test
        try:
            seg = get_segmenter(mode, transition_matrix=dummy_tm, seed=42)
            result = seg(trace)
            lengths = [len(s) for s in result["segment_ids"]]
            print(f"Mode: {mode:12s} | segments: {len(lengths):2d} | lengths: {lengths}")
        except Exception as e:
            print(f"Mode: {mode:12s} | ERROR: {e}")