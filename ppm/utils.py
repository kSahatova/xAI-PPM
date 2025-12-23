import os
import argparse
import pandas as pd
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader

from imblearn.under_sampling import RandomUnderSampler

from skpm.event_logs.split import unbiased
from ppm.datasets.event_logs import EventLog
from ppm.datasets import DatasetColumnSchema

from ppm.models.config import FreezingConfig
from sklearn.preprocessing import StandardScaler

from skpm.feature_extraction import TimestampExtractor


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

    """ if fine-tuning """
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
    if include_labels:
        df = df.loc[
            :, ["case:concept:name", "concept:name", "time:timestamp", "outcome"]
        ]
    else:
        df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]
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


def prepare_simbank_data(
    df: pd.DataFrame,
    numerical_features: List[str],
    unbiased_split_params: dict = {'test_ratio': 0.2},
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
    test_ratio = unbiased_split_params['test_ratio']
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


def calculate_accuracy(model: torch.nn.Module, data_loader: DataLoader, device: str, task='outcome_prediction'):
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
