import torch
import numpy as np
import pandas as pd
import os.path as osp
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt

from skpm import event_logs

from global_xai.map import Explainer
from global_xai.map import ConceptProperties

from ppm.datasets import DatasetSchemas
from ppm.utils import parse_args, add_outcome_labels
from setup_experiment import setup_dataloaders, setup_model


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


config_path = r'D:\PycharmProjects\xAI-PPM\configs\explain_lstm_args_for_op.txt'
checkpoints_path = r'D:\PycharmProjects\xAI-PPM\persisted_models\suffix\BPI17_rnn_outcome_bpi17.pth'

args = parse_args(config_path=config_path)
config = vars(args)
config["log"] = args.dataset
config["continuous_features"]  = (NUMERICAL_FEATURES
        if (
            args.continuous_features is not None
            and "all" in args.continuous_features
        )
        else args.continuous_features
    )
config["checkpoint_path"] = checkpoints_path
config['batch_size'] = 128


log = getattr(event_logs, config["log"])()

labels_dict = {"O_Accepted": 0, "O_Cancelled": 1, "O_Refused": 2}
column_schema = getattr(DatasetSchemas, config["log"])()
labeled_df = add_outcome_labels(log.dataframe, column_schema, labels_dict)

# Remove O_Refused to convert the task to a binary classification
binary_labeled_df = labeled_df[labeled_df["outcome"] != 2]
train_loader, test_loader = setup_dataloaders(config, binary_labeled_df, log.unbiased_split_params)
model = setup_model(config, train_loader.dataset.log, model_name='outcome_predictor')


all_features = train_loader.dataset.log.features
total_features_num  = len(all_features.categorical) + len(all_features.numerical)

trace_lens = [len(trace) for trace in train_loader.dataset.traces]
max_len =  max(trace_lens)

output_dir = Path('D:/PycharmProjects/xAI-PPM/output')
explainer_name = 'map_explainer'

output_dir_ex = output_dir / explainer_name
output_dir_ex.mkdir(parents=True, exist_ok=True)

n_concepts = 4
epochs = 10

exp = Explainer(input_dim=total_features_num,
                cat_vocab_size=27,
                output_directory=output_dir_ex,
                n_concepts=n_concepts,
                latent_dim=n_concepts * 5,
                epochs=epochs,
                batch_size=32,
                kwargs={'num_layers': 2,
                        'dropout': 0.1,
                        'hiddem_dim': 64})

exp.fit_explainer(classifier=model,
                  dataloader=train_loader)

print("Explainer training completed.")
