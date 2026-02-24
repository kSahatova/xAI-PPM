import numpy as np
from typing import Tuple, List
from timeshap.utils import calc_avg_event, calc_avg_sequence


RANDOM_SEED = 42


def build_average_event_baseline(
    train_loader, config: dict, max_bg_trace_length: int = 20, bg_sample_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average event and average sequence from random background
    traces (marginal baseline).

    Returns
    -------
    avg_event : (1, n_features)
    avg_sequence : (max_len, n_features)
    """
    np.random.seed(RANDOM_SEED)

    bc_indices = np.random.randint(0, 1000, size=bg_sample_size).tolist()

    bc_data = []
    for i in bc_indices:
        trace = train_loader.dataset[i]
        arr = np.concatenate([trace[0].numpy(), trace[1].numpy()], axis=-1)
        bc_data.append(np.expand_dims(arr, 0))

    short = [t for t in bc_data if t.shape[1] <= max_bg_trace_length]
    max_len = max(t.shape[1] for t in short)
    padded = [np.pad(t, ((0, 0), (0, max_len - t.shape[1]), (0, 0))) for t in short]
    bc_array = np.concatenate(padded, axis=0)

    cat_feats: List[int] = [0]
    num_feats: List[int] = list(range(1, 1 + len(config["continuous_features"])))
    all_names = list(config["categorical_features"]) + list(
        config["continuous_features"]
    )

    avg_sequence = calc_avg_sequence(
        bc_array,
        categorical_feats=cat_feats,
        numerical_feats=num_feats,
        model_features=all_names,
    )
    avg_event = calc_avg_event(
        avg_sequence[np.where(avg_sequence[:, 0] != 0)],
        categorical_feats=cat_feats,
        numerical_feats=num_feats,
        model_features=all_names,
    )

    return avg_event, avg_sequence
