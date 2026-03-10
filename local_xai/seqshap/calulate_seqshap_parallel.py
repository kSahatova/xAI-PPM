
import os
import numpy as np

from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from local_xai.seqshap import (
    SeqShapKernel,
    plot_segment_level_sv,
)


def _shap_one_case(
    i: int,
    case,
    seg_info: Dict[str, list],
    fetching_f,
    baseline: np.ndarray,
    output_dir: str,
    save_plots: bool,
) -> tuple[int, Dict[str, Any]]:
    seg_ids = seg_info["segment_ids"]
    seg_explainer = SeqShapKernel(
        fetching_f, baseline, rs=52, mode="segment", segment_ids=seg_ids,
    )
    seg_sv = seg_explainer.shap_values(case)
    seg_names = [f"Segment:  {j + 1}" for j in range(seg_sv.shape[0])]

    if save_plots:
        # matplotlib is not thread-safe; guard plot creation with a lock
        with _PLOT_LOCK:
            plot_segment_level_sv(seg_sv, seg_explainer, seg_names, i, output_dir)

    return i, {
        "segment_sv": seg_sv,
        "segment_names": seg_names,
        "segment_ids": seg_ids,
        "base_value": seg_explainer.expected_value,
    }



def compute_segment_shap_values_parallel(
    fetching_f,
    explicands: List,
    segments: List[Dict[str, list]],
    baseline: np.ndarray,
    config: dict,
    *,
    output_dir: str = "",
    save_plots: bool = True,
    n_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Compute segment-level SHAP values in parallel using threads.

    ThreadPoolExecutor is used (not processes) because fetching_f is a lambda
    and PyTorch releases the GIL during forward passes, so threads genuinely
    overlap on compute.

    Args:
        n_workers: number of threads. Defaults to min(32, n_cases).
                   Set to 1 to disable parallelism for debugging.
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    n = len(explicands)
    workers = min(n_workers or 32, n)
    results: List[Optional[Dict[str, Any]]] = [None] * n

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_shap_one_case, i, case, seg_info, fetching_f,
                        baseline, output_dir, save_plots): i
            for i, (case, seg_info) in enumerate(zip(explicands, segments))
        }
        for future in tqdm(as_completed(futures), total=n, desc="SHAP values"):
            i, result = future.result()
            results[i] = result

    return results