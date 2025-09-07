from sympy import group
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from typing import Tuple

from ppm.datasets.event_logs import EventLog
from sklearn.preprocessing import StandardScaler

# from skpm.event_logs.split import unbiased
from skpm.feature_extraction import TimestampExtractor

from skpm.config import EventLogConfig as elc
from skpm.event_logs.base import TUEventLog


def parse_args(
    config_path: str = "@configs/explain_lstm_args.txt",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--dataset", type=str, default="BPI17")
    parser.add_argument("--project_name", type=str, default="multi-task-icpm")
    parser.add_argument("--device", type=str, default="cuda")

    """ features and tasks """
    # e.g.: python main --categorical_features a b
    parser.add_argument("--categorical_features", nargs="+", default=["activity"])
    parser.add_argument("--categorical_targets", nargs="+", default=["activity"])
    parser.add_argument("--continuous_features", nargs="+", default="all")
    parser.add_argument("--continuous_targets", nargs="+", default=["remaining_time"])

    """ in layer config """
    parser.add_argument(
        "--strategy", type=str, default="sum", choices=["sum", "concat"]
    )

    """ model config """
    parser.add_argument(
        "--backbone",
        type=str,
        default="rnn",
        choices=["gpt2", "llama32-1b", "llama2-7b", "qwen25-05b", "rnn", "pm-gpt2"],
    )
    # if rnn
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument(
        "--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint_path", type=str, default="")

    return parser.parse_args([config_path])


def get_model_config(train_log: EventLog, config: dict):
    backbone_hf_name = "rnn"
    return {
        "embedding_size": config["embedding_size"],
        "categorical_cols": train_log.features.categorical,
        "categorical_sizes": train_log.categorical_sizes,
        "numerical_cols": train_log.features.numerical,
        "categorical_targets": train_log.targets.categorical,
        "numerical_targets": train_log.targets.numerical,
        "padding_idx": train_log.special_tokens["<PAD>"],
        "strategy": config["strategy"],
        "backbone_name": backbone_hf_name,
        "backbone_pretrained": False,
        "backbone_finetuning": config.get("fine_tuning", None),
        "backbone_type": config.get("rnn_type", None),
        "backbone_hidden_size": config["hidden_size"],
        "backbone_n_layers": config.get("n_layers", None),
        "device": config["device"],
    }


def prepare_data(
    df: pd.DataFrame, unbiased_split_params: dict, NUMERICAL_FEATURES: list
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]] #"OfferID",
    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp", ]) # "case:concept:name", "OfferID",
    train, test = unbiased(df, **unbiased_split_params)

    time_unit = "d"
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit=time_unit,
    )
    train[ts.get_feature_names_out()] = ts.fit_transform(train)
    test[ts.get_feature_names_out()] = ts.transform(test)

    train = train.drop(columns=["time:timestamp"])
    test = test.drop(columns=["time:timestamp"])

    train = train.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )
    test = test.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )

    sc = StandardScaler()
    columns = NUMERICAL_FEATURES + ["remaining_time"]
    # columns = ["accumulated_time", "remaining_time"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    return train, test


def calculate_accuracy(model: torch.nn.Module, data_loader: DataLoader, device: str):
    """
    calculates accuracy of the provided model on the given data loader
    """
    model.eval()
    total_targets = 0
    accuracy = 0

    with torch.inference_mode():
        for items in data_loader:
            x_cat, x_num, y_cat, y_num = items
            x_cat, x_num, y_cat, y_num = (
                x_cat.to(device),
                x_num.to(device),
                y_cat.to(device),
                y_num.to(device),
            )

            attention_mask = (x_cat[..., 0] != 0).long()
            total_targets += attention_mask.sum().item()

            # with torch.autocast(device_type=device, dtype=torch.float16):
            out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)

            mask = attention_mask.bool().view(-1)
            for ix, target in enumerate(data_loader.dataset.log.targets.categorical):
                predictions = torch.argmax(out[target], dim=-1)
                accuracy += (
                    (predictions.view(-1)[mask] == y_cat[..., ix].view(-1)[mask])
                    .sum()
                    .item()
                )
                # accuracy += acc

    print("Accuracy of the model: {:.3%}".format(accuracy / total_targets))



def _bounded_dataset(
    dataset: pd.DataFrame, start_date, end_date: int
) -> pd.DataFrame:
    grouped = dataset.groupby(elc.case_id, as_index=False)[elc.timestamp].agg(
        ["min", "max"]
    )

    start_date = (
        pd.Period(start_date)
        if start_date
        else dataset[elc.timestamp].min().to_period("M")
    )
    end_date = (
        pd.Period(end_date)
        if end_date
        else dataset[elc.timestamp].max().to_period("M")
    )
    bounded_cases = grouped[
        (grouped["min"].dt.to_period("M") >= start_date)
        & (grouped["max"].dt.to_period("M") <= end_date)
    ][elc.case_id].values
    dataset = dataset[dataset[elc.case_id].isin(bounded_cases)]
    return dataset


def _unbiased(dataset: pd.DataFrame, max_days: int) -> pd.DataFrame:
    grouped = (
        dataset.groupby(elc.case_id, as_index=False)[elc.timestamp]
        .agg(["min", "max"])
        .assign(
            duration=lambda x: (x["max"] - x["min"]).dt.total_seconds()
            / (24 * 60 * 60)
        )
    )

    # condition 1: cases are shorter than max_duration
    condition_1 = grouped["duration"] <= max_days * 1.00000000001
    # condition 2: drop cases starting after the dataset's last timestamp - the max_duration
    latest_start = dataset[elc.timestamp].max() - pd.Timedelta(
        max_days, unit="D"
    )
    condition_2 = grouped["min"] <= latest_start

    unbiased_cases = grouped[condition_1 & condition_2][elc.case_id].values
    dataset = dataset[dataset[elc.case_id].isin(unbiased_cases)]
    return dataset


def unbiased(
    dataset: pd.DataFrame | TUEventLog,
    start_date: str | pd.Period | None,
    end_date: str | pd.Period | None,
    max_days: int,
    test_len: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unbiased split of event log into training and test set [1].

    Code adapted from [2].

    Parameters
    ----------
    dataset: pd.DataFrame
        Event log.

    start_date: str
        Start date of the event log.

    end_date: str
        End date of the event log.

    max_days: int
        Maximum duration of cases.

    test_len: float, default=0.2
        Proportion of cases to be used for the test set.

    Returns
    -------
    - df_train: pd.DataFrame, training set
    - df_test: pd.DataFrame, test set

    Example
    -------
    >>> from skpm.event_logs import BPI12
    >>> from skpm.event_logs import split
    >>> bpi12 = BPI12()
    >>> df_train, df_test = split.unbiased(bpi12, **bpi12.unbiased_split_params)
    >>> df_train.shape, df_test.shape
    ((117546, 7), (55952, 7))

    References:
    -----------
    [1] Hans Weytjens, Jochen De Weerdt. Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring, 2021. doi: 10.1007/978-3-030-94343-1_2
    [2] https://github.com/hansweytjens/predictive-process-monitoring-benchmarks
    """
    if isinstance(dataset, TUEventLog):
        dataset = dataset.dataframe
        
    dataset = dataset.copy()
    
    dataset[elc.timestamp] = pd.to_datetime(
        dataset[elc.timestamp], utc=True
    ).dt.tz_localize(None)

    # bounding the event log
    if start_date or end_date:
        dataset = _bounded_dataset(dataset, start_date, end_date)

    # drop longest cases and debiasing end of dataset
    dataset = _unbiased(dataset, max_days)

    # preliminaries
    grouped = dataset.groupby(elc.case_id, as_index=False)[elc.timestamp].agg(
        ["min", "max"]
    )

    ### TEST SET ###
    first_test_case_nr = int(len(grouped) * (1 - test_len))
    first_test_start_time = (
        grouped["min"].sort_values().values[first_test_case_nr]
    )
    # retain cases that end after first_test_start time
    test_case_nrs = grouped.loc[
        grouped["max"].values >= first_test_start_time, elc.case_id
    ]
    df_test = dataset[dataset[elc.case_id].isin(test_case_nrs)].reset_index(
        drop=True
    )

    #### TRAINING SET ###
    df_train = dataset[~dataset[elc.case_id].isin(test_case_nrs)].reset_index(
        drop=True
    )

    # get prefixes of the train and test sets 

    def extract_prefix(group, prefix_len=25):
        return group.head(prefix_len)

    case_lengths_train = df_train.groupby('case:concept:name', as_index=False).size()
    df_train_w_case_len = df_train.merge(case_lengths_train, on='case:concept:name', how='left')
    # df_train = df_train_w_case_len[df_train_w_case_len['size']<= 25].drop('size', axis=1)
    df_train.groupby('case:concept:name').apply(extract_prefix).reset_index(drop=True)

    case_lengths_test = df_test.groupby('case:concept:name', as_index=False).size()
    df_test_w_case_len = df_test.merge(case_lengths_test, on='case:concept:name', how='left')
    # df_test = df_test_w_case_len[df_test_w_case_len['size']<= 25].drop('size', axis=1)
    df_train.groupby('case:concept:name').apply(extract_prefix).reset_index(drop=True)

    return df_train, df_test

