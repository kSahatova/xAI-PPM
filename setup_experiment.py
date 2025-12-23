import torch
import numpy as np
import pandas as pd

from typing import List
from torch.utils.data import DataLoader

from ppm.datasets import ContinuousTraces
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets
from ppm.datasets.utils import continuous
from ppm import models as ppm_models
from ppm.engine.utils import load_checkpoint
from ppm.utils import prepare_data, prepare_simbank_data, get_model_config


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
    if config["log"] == "BPI17":
        result = prepare_data(
            log,
            unbiased_split_params,
            NUMERICAL_FEATURES,
            include_labels=True,
        )
        # prepare_data may return either (train, test) or (train, test, train_timestamps, test_timestamps)
        if isinstance(result, tuple) and len(result) == 4:
            train, test, train_timestamps, test_timestamps = result
        elif isinstance(result, tuple) and len(result) == 2:
            train, test = result
        else:
            raise ValueError("Unexpected return value from prepare_data()")
    elif config["log"] == "synthetic":
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
        name=config["log"],
    )

    test_log = EventLog(
        dataframe=test,
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=False,
        name=config["log"],
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    return train_loader, test_loader


def setup_model(config: dict, log: EventLog, model_name: str = "outcome_predictor"):
    # Loading a pre-trained model
    model_config = get_model_config(log, config)
    if model_name == "outcome_predictor":
        model_config.pop("categorical_targets", None)
        model_config.pop("numerical_targets", None)

    model_class_name = ("").join([el.capitalize() for el in model_name.split("_")])
    model_class = getattr(ppm_models, model_class_name)
    model = model_class(**model_config).to(device=config["device"])

    ckpt = load_checkpoint(config["checkpoint_path"], map_location=config["device"])
    if isinstance(ckpt, dict) and "net" in ckpt.keys():
        ckpt = ckpt["net"]
    model.load_state_dict(ckpt)

    return model


