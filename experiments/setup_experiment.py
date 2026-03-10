from typing import OrderedDict, Tuple, Optional

import torch
import pandas as pd

from torch.utils.data import DataLoader

from skpm import event_logs
from ppm.datasets import DatasetSchemas
from ppm.datasets import ContinuousTraces
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets
from ppm.datasets.utils import continuous
from ppm import models as ppm_models
from ppm.engine.utils import load_checkpoint
from ppm.utils import (
    prepare_data,
    prepare_simbank_data,
    get_model_config,
    parse_args,
    add_outcome_labels,
)


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

NUMERICAL_FEATURES = [
    "accumulated_time",
    "day_of_month",
    "day_of_week",
    "day_of_year",
    "hour_of_day",
    "min_of_hour",
    "month_of_year",
    "sec_of_min",
    "secs_within_day",
    "week_of_year",
]


def setup_dataloaders(config: dict, log: pd.DataFrame, unbiased_split_params):
    
    train_timestamps, test_timestamps = None, None
    
    if config["dataset"] == "BPI17":
        result = prepare_data(
            log,
            unbiased_split_params,
            NUMERICAL_FEATURES,
            include_labels=True,
            return_timestamps=True
        )
        # prepare_data may return either (train, test) or (train, test, train_timestamps, test_timestamps)
        if isinstance(result, tuple) and len(result) == 4:
            train, test, train_timestamps, test_timestamps = result
        elif isinstance(result, tuple) and len(result) == 2:
            train, test = result
        else:
            raise ValueError("Unexpected return value from prepare_data()")
    elif config["dataset"] == "synthetic":
        train, test = prepare_simbank_data(log, NUMERICAL_FEATURES)
    else:
        raise ValueError(
            f"The data preprocessing function has not been implemented for the log {config['log']}"
        )

    event_features = EventFeatures(
        categorical=config["categorical_features"],
        numerical=config["continuous_features"],
    )
    event_targets = EventTargets(
        categorical=config["categorical_targets"],
        numerical=config["continuous_targets"],
    )

    train_log = EventLog(
        dataframe=train,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=True,
        name=config["dataset"],
    )

    test_log = EventLog(
        dataframe=test,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=False,
        name=config["dataset"],
        vocabs=train_log.get_vocabs(),
    )

    dataset_device = (
        config["device"]
        if config["backbone"] not in ["gpt2", "llama32-1b", "llama2-7b", "qwen25-05b"]
        else "cpu"
    )

    train_dataset = ContinuousTraces(
        log=train_log,
        refresh_cache=True,
        device=dataset_device,
    )
    test_dataset = ContinuousTraces(
        log=test_log,
        refresh_cache=True,
        device=dataset_device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    if (train_timestamps is not None) and (test_timestamps is not None):
        train_dataset.timestamps = train_timestamps
        test_dataset.timestamps = test_timestamps

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    return train_loader, test_loader


def setup_model(
    config: dict,
    log: EventLog,
    checkpoint_path: str,
    model_name: str = "outcome_predictor",
):
    # Loading a pre-trained model
    model_config = get_model_config(log, config)
    if model_name == "outcome_predictor":
        model_config.pop("categorical_targets", None)
        model_config.pop("numerical_targets", None)

    model_class_name = ("").join([el.capitalize() for el in model_name.split("_")])
    model_class = getattr(ppm_models, model_class_name)
    model = model_class(**model_config).to(device=config["device"])

    ckpt = load_checkpoint(checkpoint_path, map_location=config["device"])
    if isinstance(ckpt, dict) and "net" in ckpt.keys():
        ckpt = ckpt["net"]
    model.load_state_dict(ckpt)

    return model


def create_loader_from_dataframe(df: pd.DataFrame, config: dict, cf_vocab: Tuple[OrderedDict, OrderedDict]):
    
    event_features = EventFeatures(
        categorical=config["categorical_features"],
        numerical=config["continuous_features"],
    )
    event_targets = EventTargets(
        categorical=config["categorical_targets"],
        numerical=config["continuous_targets"],
    )

    event_log = EventLog(
        dataframe=df,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=False,
        name=config["dataset"],
        vocabs=cf_vocab,
    )

    dataset = ContinuousTraces(
        log=event_log,
        refresh_cache=True,
        device=config['device'],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    return data_loader


def load_data_and_model(config_path: str, checkpoint_path: str):
    """Load event log, prepare binary-labelled dataframe, build data loaders
    and load the pre-trained model.

    Returns
    -------
    config : dict
    train_loader, test_loader : DataLoader
    model : nn.Module  (already in eval mode)
    """
    torch.manual_seed(RANDOM_SEED)
    LABELS_DICT = {"O_Accepted": 0, "O_Cancelled": 1, "O_Refused": 2}

    args = parse_args(config_path=config_path)
    config = vars(args)
    config["continuous_features"] = (
        NUMERICAL_FEATURES
        if config["continuous_features"] == "all"
        else config["continuous_features"]
    )

    log = getattr(event_logs, config["dataset"])()
    column_schema = getattr(DatasetSchemas, config["dataset"])()
    labeled_df = add_outcome_labels(log.dataframe, column_schema, LABELS_DICT)
    binary_labeled_df = labeled_df[labeled_df["outcome"] != 2]  # drop O_Refused

    train_loader, test_loader = setup_dataloaders(
        config, binary_labeled_df, log.unbiased_split_params
    )
    model = setup_model(
        config,
        train_loader.dataset.log,
        checkpoint_path,
        model_name="outcome_predictor",
    )
    model.eval()

    return config, train_loader, test_loader, model
