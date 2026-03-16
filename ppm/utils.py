import os
import argparse
import pandas as pd
from typing import List, Optional, Union

import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler

from skpm.event_logs.split import temporal, unbiased
from ppm.build.lib.metrics import tracker
from ppm.datasets.event_logs import EventLog
from ppm.datasets import DatasetColumnSchema

from ppm.models.config import FreezingConfig
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from skpm.feature_extraction import TimestampExtractor
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use("seaborn-v0_8-whitegrid")


def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def experiment_exists(config: dict):
    """Check if experiment with the same config exists."""

    experiments = get_existing_experiments()
    if experiments is None:
        return False

    # TODO: make this drop generic
    experiments.drop(columns=["id", "n_features"], inplace=True, errors="ignore")
    # experiments.fillna(value=None, axis=0, inplace=True)  # none is not valid
    # grad_clip is None for config and nan for exp
    for _, exp in experiments.T.to_dict().items():
        exp["grad_clip"] = None
        if exp == config:
            return True
    return False


def get_existing_experiments(force_fetch=False, project="cosmo-ltl"):
    if os.path.exists("experiments.csv") and not force_fetch:
        try:
            return pd.read_csv("experiments.csv")
        except:
            return None
    else:
        return fetch_experiments(project)


def fetch_experiments(project="cosmo-v4"):
    try:
        import wandb
    except ImportError:
        print("wandb not installed")
        return

    api = wandb.Api()
    runs = api.runs("raseidi/" + project)
    metrics = [
        "train_a_loss",
        "train_a_acc",
        "train_t_loss",
        "test_a_loss",
        "test_a_acc",
        "test_t_loss",
        "_runtime",
    ]

    experiments = pd.DataFrame()
    for r in runs:
        if r.state != "finished":
            continue

        new = pd.DataFrame([r.config])
        new["id"] = r.id
        new["name"] = r.name

        for m in metrics:
            new[m] = r.summary[m]
        experiments = pd.concat((experiments, new), ignore_index=True)

    experiments.reset_index(inplace=True, drop=True)
    experiments.to_csv("experiments.csv", index=False)
    return experiments


def extract_timestamp_features(group):
    """extract the time since the last event in minutes"""
    group = group.sort_values("time:timestamp", ascending=False, kind='mergesort')
    
    tmp = group["time:timestamp"] - group["time:timestamp"].shift(-1)
    tmp = tmp.fillna(pd.Timedelta(0))
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes
    group = group.sort_values("time:timestamp", ascending=True, kind='mergesort')
    return group


def check_if_activity_exists_and_time_less_than(group, activity):
    relevant_activity_idxs = np.where(group["concept:name"] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        if group["timesincelastevent"].iloc[idx] <= 28 * 1440: # return in less than 28 days
            group["outcome"] = 1
            return group[:idx]
        else:
            group["outcome"] = 0
            return group[:idx]
    else:
        group["outcome"] = 0
        return group
    

def parse_args(config_path: str = ""):
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument("--dataset", type=str, default="BPI17")
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--persist_model", action="store_true", default=False)
    parser.add_argument("--project_name", type=str, default="multi-task-icpm")

    """ training config """
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    """ features and tasks """
    # e.g.: python main --categorical_features a b
    parser.add_argument("--categorical_features", nargs="+", default=["activity"])
    parser.add_argument("--categorical_targets", nargs="+", default=None)
    parser.add_argument("--continuous_features", nargs="+", default="all")
    parser.add_argument("--continuous_targets", nargs="+", default=None)

    """ in layer config """
    parser.add_argument(
        "--strategy", type=str, default="sum", choices=["sum", "concat"]
    )
    parser.add_argument(
        "--pos_encoding_form",
        type=str,
        default=None,
        choices=["sinusoidal", "learnable", "random", "dummy"],
    )
    parser.add_argument(
        "--pos_encoding_strategy", type=str, default="sum", choices=["sum", "concat"]
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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

    """if fine-tuning """
    parser.add_argument(
        "--fine_tuning", type=str, default=None, choices=["lora", "freeze"]
    )
    # if lora
    parser.add_argument("--r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    # if freeze
    parser.add_argument(
        "--freeze_layers",
        nargs="+",
        type=int,
        default=None,
        help="List of layer indices to freeze. If None, all layers are frozen.",
    )

    return parser.parse_args([f"@{config_path}"])


def add_outcome_labels(
    log_df: pd.DataFrame,
    col_schema: DatasetColumnSchema,
    labels_dict: dict,
    resource_freq_threshold: int = 10,
    max_category_levels: int = 10,
) -> pd.DataFrame:
    """
    Adds labels according to the provided dictionary. Labels are assigned to cases based on the last activity
    """
    # Create label mapping once
    dt_labeled = log_df.copy()
    relevant_offer_events = list(labels_dict.keys())

    timestamp = col_schema.timestamp_col
    activity = col_schema.activity_col
    case_id = col_schema.case_id_col
    resource = col_schema.resource_col
    label_col = col_schema.label_col
    cat_cols = col_schema.cat_cols

    # Optimize: Filter once, use stable sort if needed, chain operations
    last_o_events = (
        dt_labeled[dt_labeled.loc[:, "EventOrigin"] == "Offer"]
        .sort_values(timestamp, ascending=True, kind="mergesort")
        .groupby(case_id)[activity]
        .last()
        .rename("last_o_activity")
    )

    dt_labeled = dt_labeled.merge(
        last_o_events, left_on=case_id, right_index=True, how="inner"
    )
    dt_labeled = dt_labeled[dt_labeled.last_o_activity.isin(relevant_offer_events)]

    dt_labeled[label_col] = dt_labeled["last_o_activity"].map(labels_dict)
    core_cols = col_schema.static_cols + col_schema.dynamic_cols
    dt_labeled = dt_labeled.loc[:, core_cols + ["last_o_activity"]]

    # Sort once and reuse grouped object
    dt_labeled = dt_labeled.sort_values(timestamp, ascending=True, kind="mergesort")
    grouped = dt_labeled.groupby(case_id)

    # Fill missing values
    cols_to_fill = [col for col in core_cols if col in dt_labeled.columns]
    dt_labeled[cols_to_fill] = grouped[cols_to_fill].ffill()
    dt_labeled[cat_cols] = dt_labeled[cat_cols].fillna("missing")
    dt_labeled = dt_labeled.fillna(0)

    # Precompute value counts for all categorical columns
    value_counts_cache = {col: dt_labeled[col].value_counts() for col in cat_cols}

    # Set infrequent factor levels using vectorized operations
    for col in cat_cols:
        counts = value_counts_cache[col]
        if col == resource:
            # Keep only frequent resources
            frequent_values = counts[counts >= resource_freq_threshold].index
            dt_labeled.loc[~dt_labeled[col].isin(frequent_values), col] = "other"
        elif col != activity:
            # Keep only top N categories
            if len(counts) > max_category_levels:
                top_categories = counts.index[:max_category_levels]
                dt_labeled.loc[~dt_labeled[col].isin(top_categories), col] = "other"

    return dt_labeled


def prepare_data(
    df: pd.DataFrame,
    unbiased_split_params: dict,
    numerical_features: List[str],
    return_timestamps: bool = False,
    include_labels: bool = False,
):
    if not include_labels:
        df = df.loc[:, df.columns.drop("outcome")]

    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = unbiased(df, **unbiased_split_params)

    time_unit = "d"
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit=time_unit,
    )
    train[ts.get_feature_names_out()] = ts.fit_transform(train)
    test[ts.get_feature_names_out()] = ts.transform(test)

    train_timestamps = train.loc[:, "time:timestamp"]
    test_timestamps = test.loc[:, "time:timestamp"]

    train = train.drop(columns=["time:timestamp"])
    test = test.drop(columns=["time:timestamp"])

    train = train.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )
    test = test.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )

    sc = StandardScaler()
    columns = numerical_features + ["remaining_time"]
    # columns = ["accumulated_time", "remaining_time"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    if return_timestamps:
        return train, test, train_timestamps, test_timestamps

    return train, test


def prepare_sepsis_data(
    df: pd.DataFrame,
    # unbiased_split_params: dict,
    numerical_features: List[str],
    return_timestamps: bool = False,
    include_labels: bool = False,
):
    if include_labels:
        df = df.loc[
            :,
            [
                "case:concept:name",
                "concept:name",
                "time:timestamp",
                "outcome",
            ],
        ]
    else:
        df = df.loc[
            :,
            [
                "case:concept:name",
                "concept:name",
                "time:timestamp",
            ],
        ]

    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = temporal(df, test_len=0.3)

    time_unit = "d"
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit=time_unit,
    )
    train[ts.get_feature_names_out()] = ts.fit_transform(train)
    test[ts.get_feature_names_out()] = ts.transform(test)

    train_timestamps = train.loc[:, "time:timestamp"]
    test_timestamps = test.loc[:, "time:timestamp"]

    train = train.drop(columns=["time:timestamp"])
    test = test.drop(columns=["time:timestamp"])

    train = train.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )
    test = test.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )

    sc = StandardScaler()
    columns = numerical_features + ["remaining_time"]
    # columns = ["accumulated_time", "remaining_time"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    if return_timestamps:
        return train, test, train_timestamps, test_timestamps

    return train, test



def prepare_bpi15_data(
    df: pd.DataFrame,
    # unbiased_split_params: dict,
    numerical_features: List[str],
    return_timestamps: bool = False,
):
    df = df.loc[
        :,
        [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
            "outcome",
        ],
    ]

    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]

    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    train, test = temporal(df, test_len=0.3)

    time_unit = "d"
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit=time_unit,
    )
    train[ts.get_feature_names_out()] = ts.fit_transform(train)
    test[ts.get_feature_names_out()] = ts.transform(test)

    train_timestamps = train.loc[:, "time:timestamp"]
    test_timestamps = test.loc[:, "time:timestamp"]

    train = train.drop(columns=["time:timestamp"])
    test = test.drop(columns=["time:timestamp"])

    train = train.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )
    test = test.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )

    sc = StandardScaler()
    columns = numerical_features + ["remaining_time"]
    # columns = ["accumulated_time", "remaining_time"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    if return_timestamps:
        return train, test, train_timestamps, test_timestamps

    return train, test


def prepare_simbank_data(
    df: pd.DataFrame,
    numerical_features: List[str],
    unbiased_split_params: dict = {"test_ratio": 0.2},
):
    # Extract last activity of each case for the outcome feature generation
    last_activities = (
        df.sort_values(["case_nr", "timestamp"], ascending=True)
        .groupby("case_nr")["activity"]
        .last()
    )

    # Outcome labels : 0 - cancel_application , 1 - receive_acceptance
    outcome = (last_activities == "receive_acceptance").astype(int).rename("outcome")
    labeled_df = df.merge(outcome, on="case_nr", how="left")
    labeled_df = labeled_df.sort_values(by=["case_nr", "timestamp"])

    # Balance the dataset
    case_nr_outcome = labeled_df.groupby("case_nr", as_index=False)["outcome"].last()
    rus = RandomUnderSampler(random_state=42)
    case_nr_res, _ = rus.fit_resample(
        case_nr_outcome["case_nr"].values.reshape(-1, 1),  # type: ignore
        case_nr_outcome["outcome"].values,
    )
    balanced_df = labeled_df[labeled_df["case_nr"].isin(case_nr_res.squeeze())]

    # Reduce number of features
    included_features = [
        "case_nr",
        "activity",
        "amount",
        "unc_quality",
        "est_quality",
        "timestamp",
        "interest_rate",
        "discount_factor",
        "elapsed_time",
        "outcome",
    ]
    reduced_df = balanced_df.loc[:, included_features]
    reduced_df = reduced_df.sort_values(by=["case_nr"])

    grouped = reduced_df.groupby("case_nr", as_index=False)["timestamp"].agg(
        ["min", "max"]
    )

    # Splitting into train and test
    ### TEST SET ###
    test_ratio = unbiased_split_params["test_ratio"]
    first_test_case_nr = int(len(grouped) * (1 - test_ratio))
    first_test_start_time = grouped["min"].sort_values().values[first_test_case_nr]
    # retain cases that end after first_test_start time
    test_case_nrs = grouped.loc[
        grouped["max"].values >= first_test_start_time, "case_nr"
    ]
    test = (
        reduced_df[reduced_df["case_nr"].isin(test_case_nrs)]
        .sort_values(by=["case_nr", "timestamp"])
        .reset_index(drop=True)
    )

    #### TRAINING SET ###
    train = (
        reduced_df[~reduced_df["case_nr"].isin(test_case_nrs)]
        .sort_values(by=["case_nr", "timestamp"])
        .reset_index(drop=True)
    )

    train = train.drop(columns=["timestamp"], axis=1)
    test = test.drop(columns=["timestamp"], axis=1)

    # Filling in missing values
    train["interest_rate"] = train["interest_rate"].fillna(0.0)
    test["interest_rate"] = test["interest_rate"].fillna(0.0)
    train["discount_factor"] = train["discount_factor"].fillna(0.0)
    test["discount_factor"] = test["discount_factor"].fillna(0.0)

    # Scaling numeric features
    sc = StandardScaler()
    train.loc[:, numerical_features] = sc.fit_transform(train[numerical_features])
    test.loc[:, numerical_features] = sc.transform(test[numerical_features])

    return train, test


def get_fine_tuning(fine_tuning, **kwargs):
    if fine_tuning == "lora":
        target_modules = (
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "up_proj",
                "down_proj",
                "o_proj",
                "gate_proj",
            ]
            if "gpt2" not in kwargs["model"]
            else None
        )
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=kwargs["r"],
            lora_alpha=kwargs["lora_alpha"],
            target_modules=target_modules,
            use_rslora=True,
        )
    elif fine_tuning == "freeze":
        return FreezingConfig(
            ix_layers=kwargs["freeze_layers"],
            module_path=kwargs["fine_tuning_module_path"],
        )
    elif fine_tuning is None:
        return
    else:
        raise ValueError("Invalid fine-tuning strategy")


def get_model_config(
    log: EventLog, config: dict, pretrained_configs: Optional[dict] = {}
):
    pretrained_config = pretrained_configs.get(config["backbone"], {})
    if pretrained_config:
        fine_tuning = get_fine_tuning(
            fine_tuning=config["fine_tuning"],
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            freeze_layers=config["freeze_layers"],
            fine_tuning_module_path=pretrained_config["fine_tuning_module_path"],
            model=config["backbone"],
        )
        pretrained_config["fine_tuning"] = fine_tuning
    if config["backbone"] != "rnn":
        backbone_hf_name = pretrained_config["name"]
    else:
        backbone_hf_name = "rnn"
    return {
        "embedding_size": config["embedding_size"],
        "categorical_cols": log.features.categorical,
        "categorical_sizes": log.categorical_sizes,
        "numerical_cols": log.features.numerical,
        "categorical_targets": log.targets.categorical,
        "numerical_targets": log.targets.numerical,
        "padding_idx": log.special_tokens["<PAD>"],
        "strategy": config["strategy"],
        "pos_encoding_form": config.get("pos_encoding_form", None),
        "pos_encoding_strategy": config.get("pos_encoding_strategy", ""),
        "backbone_name": backbone_hf_name,
        "backbone_pretrained": True if pretrained_config else False,
        "backbone_finetuning": pretrained_config.get("fine_tuning", None),
        "backbone_type": config.get("rnn_type", None),
        "backbone_hidden_size": config["hidden_size"],
        "backbone_n_layers": config.get("n_layers", None),
        "device": config["device"],
    }


def calculate_accuracy(model: torch.nn.Module, data_loader: DataLoader, device: str):
    """
    Calculates accuracy of the provided model on the given data loader
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

            out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)

            predictions = ((out.squeeze(-1)) > 0.5).float()
            acc = (
                (
                    predictions.squeeze(-1)[attention_mask.bool()]
                    == y_cat.squeeze(-1)[attention_mask.bool()]
                )
                .sum()
                .item()
            )
            accuracy += acc
    print("Accuracy of the model: {:.3%}".format(accuracy / total_targets))


def extract_one_offer_cases(trace_set, o_created_ind=15):
    """Extracts case ids with one and multiple offers"""
    one_offer_ids = []
    multiple_offers_ids = {}
    for i, trace in enumerate(trace_set):
        offered_times = np.where(trace == o_created_ind)[0].size
        if offered_times == 1:
            one_offer_ids.append(i)
        else:
            multiple_offers_ids[i] = offered_times
    return one_offer_ids, multiple_offers_ids


def extract_explicands_samples(model: torch.nn.Module, dataloader: DataLoader, 
                               prefix_len: int=15, explicands_num: Union[int, None]=10, 
                               threshold: float=0.5, one_offer_cases: bool=True):
    """
    Extracting the prefixes of cases that will be explained in the experiments,
    including true positives and negatives, and false positives and negatives.
    Args:
        model: The trained outcome prediction model.
        test_loader: The dataloader for the test set.
        prefix_len: The length of the prefixes to be extracted.
        explicands_num: The number of explicands to be extracted for each group (tp, tn, fp, fn).
    """

    pred_cases_info = {
        "tp": {"ids": [], "cases": [], "y_pred": [], "y_true": []},
        "tn": {"ids": [], "cases": [], "y_pred": [], "y_true": []},
        "fp": {"ids": [], "cases": [], "y_pred": [], "y_true": []},
        "fn": {"ids": [], "cases": [], "y_pred": [], "y_true": []},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: change the "if"
    ds_name = dataloader.dataset.log.name.lower()

    for ind, batch in enumerate(dataloader):
        x_cat, x_num, y_cat, _ = batch
        if x_cat.shape[1] < prefix_len:
            continue
        if ds_name == 'bpi17':
            if 14 in x_cat[0, :prefix_len, 0] or 13 in x_cat[0, :prefix_len, 0]:
                continue
        elif ds_name == 'sepsis':
            if 18 in x_cat[0, :prefix_len, 0]:
                continue
        x_cat, x_num, y_cat = (
            x_cat[:, :prefix_len, :],
            x_num[:, :prefix_len, :],
            y_cat[:, :prefix_len],
        )
        x_cat, x_num = (x_cat.to(device), x_num.to(device))

        attention_mask = (x_cat[..., 0] != 0).long()
        out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
        prediction = ((out.squeeze(1)) > threshold).float()[0, -1].item()

        case = np.concatenate([x_cat.numpy(), x_num.numpy()], axis=-1)
        y_true = float(y_cat[0, -1].item())
        sample_name = ""

        # Extracting true positives and negatives
        if prediction == y_true:
            if prediction == 0:
                sample_name = "tn"
            else:
                sample_name = "tp"
        # Extracting misclassified cases
        else:
            if prediction == 1:
                sample_name = "fp"
            else:
                sample_name = "fn"

        pred_cases_info[sample_name]["ids"].append(ind)
        pred_cases_info[sample_name]["cases"].append(case)
        pred_cases_info[sample_name]["y_pred"].append(out.squeeze(1)[0, -1].item())
        pred_cases_info[sample_name]["y_true"].append(y_true)

    print(
        "TP case number ('O_Cancelled' correctly pred):",
        len(pred_cases_info["tp"]["cases"]),
    )
    print(
        "TN case number ('O_Accepted' correctly pred) :",
        len(pred_cases_info["tn"]["cases"]),
    )
    print(
        "FP case number ('O_Cancelled' for accepted cases):",
        len(pred_cases_info["fp"]["cases"]),
    )
    print(
        "FN case number ('O_Accepted' for cancelled cases):",
        len(pred_cases_info["fn"]["cases"]),
    )

    if one_offer_cases:
        explicands_w_one_offer = {}

        for sample_name in ["tp", "tn", "fp", "fn"]:
            sample_one_offer_ids, _ = extract_one_offer_cases(
                [trace[0, :, 0] for trace in pred_cases_info[sample_name]["cases"]]
            )
            indices = sample_one_offer_ids[:explicands_num]
            explicands_w_one_offer[sample_name] = {
                "cases": [pred_cases_info[sample_name]["cases"][idx] for idx in indices],
                "predictions": [
                    pred_cases_info[sample_name]["y_pred"][idx] for idx in indices
                ],
                "labels": [pred_cases_info[sample_name]["y_true"][idx] for idx in indices],
            }

        return explicands_w_one_offer
    
    return pred_cases_info


def calculate_auc(model, dataloader, device):
    
    predictions = []
    true_labels = []
    model.eval()

    with torch.inference_mode():
        for items in dataloader:
            x_cat, x_num, y_cat, y_num = items
            x_cat, x_num, y_cat, y_num = (
                x_cat.to(device),
                x_num.to(device),
                y_cat.to(device),
                y_num.to(device),
            )

            true_labels.append(y_cat[0, -1,  :])
            attention_mask = (x_cat[..., 0] != 0).long()

            out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
            predictions.append(out[0, -1, :])

    auc =  roc_auc_score(true_labels, predictions)
    print("AUC of the model: {:.3%}".format(auc))



def calculate_accuracy_per_position(
    model, dataloader, device: str = "cuda", save_path: str = None, show: bool = True
):
    """Compute and plot accuracy per prefix position.

    Returns a dict with `positions`, `accuracies` and `valid_counts` so callers
    can programmatically inspect results. If `save_path` is provided the plot
    will be saved to that path.
    """

    model.eval()

    metrics = {}
    valid_positions = {}

    with torch.inference_mode():
        for items in dataloader:
            x_cat, x_num, y_cat, y_num = items
            x_cat, x_num, y_cat, y_num = (
                x_cat.to(device),
                x_num.to(device),
                y_cat.to(device),
                y_num.to(device),
            )

            attention_mask = (x_cat[..., 0] != 0).long()

            out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
            predictions = ((out.squeeze(-1)) > 0.5).float()

            max_len = attention_mask.size(1)
            idxs = torch.arange(max_len).unsqueeze(0).to(device)  # [1, S]
            lengths = attention_mask.sum(dim=-1)  # [B]
            mask = idxs < lengths.unsqueeze(1)  # [B, S]

            correct = (predictions.squeeze() == y_cat.squeeze()) & mask
            for t in range(max_len):
                valid = mask[:, t]
                if valid.any():
                    acc_t = correct[:, t][valid].float().sum()
                else:
                    acc_t = torch.tensor(float("nan"))
                metrics.setdefault(f"acc_pos_{t}", 0.0)
                metrics[f"acc_pos_{t}"] += acc_t.item()
                valid_positions.setdefault(f"acc_pos_{t}", 0)
                valid_positions[f"acc_pos_{t}"] += int(valid.sum().item())

    # Build ordered lists of positions and compute accuracies (as proportions)
    pos_indices = sorted([int(k.split("_")[-1]) for k in valid_positions.keys()])
    positions = [p + 1 for p in pos_indices]  # convert to 1-based for plotting
    accuracies = []
    valid_counts = []
    for p in pos_indices:
        key = f"acc_pos_{p}"
        count = valid_positions.get(key, 0)
        valid_counts.append(count)
        if count > 0:
            acc = metrics.get(key, 0.0) / count
        else:
            acc = float("nan")
        accuracies.append(acc)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.plot(pos_indices, accuracies, color='tab:blue', label='Test Accuracy', marker="o", linestyle="-",)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Prefix Length')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per position for test set')
    plt.legend()
    plt.grid(True)

    # Annotate with counts (optional small text)
    # for x, y, c in zip(positions, accuracies, valid_counts):
    #     plt.text(x, max(0.0, y - 0.04), f"n={c}", ha="center", fontsize=8)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "positions": positions,
        "accuracies": accuracies,
        "valid_counts": valid_counts,
    }

    # import pandas as pd

    # df = pd.read_csv("a.csv")

    # plt.figure(figsize=(10, 6))
    # plt.plot(df['acc position'], df['acc_train'], label='Train Accuracy')
    # plt.plot(df['acc position'], df['acc_test'], label='Test Accuracy')
    # plt.xlabel('Accuracy Position')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy per Position for Train and Test Sets')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("accuracy_per_position.png")
