import os
import numpy as np
import pandas as pd
from pprint import pprint

import torch

from skpm.event_logs import bpi
from ppm.datasets.labeling import verify_bpi15_ltl_rule
from ppm.datasets.utils import continuous
from ppm.datasets import ContinuousTraces
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets


from ppm.engine.op import train_engine
from ppm.models import OutcomePredictor

from ppm.utils import parse_args, prepare_data, get_model_config


def main(training_config: dict):
    log = getattr(bpi, training_config["log"])(
        cache_folder=r"D:\PycharmProjects\xAI-PPM\data"
    )

    # Apply labeling function to the event log
    labels = (
        log.dataframe.groupby("case:concept:name")["concept:name"]
        .apply(verify_bpi15_ltl_rule)
        .rename("outcome")
    )
    labeled_df = log.dataframe.merge(right=labels, on="case:concept:name", how="left")

    return


if __name__ == "__main__":
    config_path = "configs/train_lstm_args_for_op_bpi15.txt"
    args = parse_args(config_path)

    training_config = {
        # args to pop before logging
        "project_name": args.project_name,
        "wandb": args.wandb,
        "persist_model": args.persist_model,
        # args to log
        "log": args.dataset,
        "device": args.device,
        # architecture
        "backbone": args.backbone,
        "rnn_type": args.rnn_type,
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        # hyperparameters
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "epochs": args.epochs,
        # fine-tuning
        "fine_tuning": args.fine_tuning,
        "r": args.r,  # LoRA
        "lora_alpha": args.lora_alpha,  # LoRA
        "freeze_layers": args.freeze_layers,  # Freeze
        # features and tasks
        "categorical_features": args.categorical_features,
        # "continuous_features": (
        #     NUMERICAL_FEATURES
        #     if (
        #         args.continuous_features is not None
        #         and "all" in args.continuous_features
        #     )
        #     else args.continuous_features
        # ),
        "categorical_targets": args.categorical_targets,
        "continuous_targets": args.continuous_targets,
        "strategy": args.strategy,
        "pos_encoding_form": args.pos_encoding_form,
        "pos_encoding_strategy": args.pos_encoding_strategy,
        "checkpoint_dir": args.checkpoint_dir,
    }

    pprint(training_config)
    print("=" * 80)
    main(training_config)
