import pprint
import pickle
import os.path as osp
import pandas as pd

import torch
from torch.utils.data import DataLoader

from ppm.datasets.utils import continuous
from ppm.datasets import ContinuousTraces
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets

from ppm.engine.op import train_engine
from ppm.models import OutcomePredictor

from ppm.utils import parse_args, get_model_config


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

NUMERIC_FEATURES = [
    "amount", 
    "unc_quality", 
    "est_quality", 
    "interest_rate",
    "discount_factor",
    "elapsed_time"
]


def main(training_config: dict):

    synt_data_dir = r"data\synthetic_data"
    data_file = "dataloan_log_['choose_procedure']_100000_train_normal.csv"
    file_path = osp.join(synt_data_dir, data_file)

    # Read a file
    with open(file_path, 'rb') as f:
        log = pickle.load(f)
    # getting rid of the  default 'outcome' used for simulation purposes 
    log = log.drop("outcome", axis=1)
    
    # Extract last activity of each case for the outcome feature generation
    last_activities = log.sort_values(["case_nr", "timestamp"], ascending=True) \
                        .groupby("case_nr")["activity"] \
                        .last()

    # Outcome labels : 0 - cancel_application , 1 - receive_acceptance 
    outcome = (last_activities == 'receive_acceptance').astype(int).rename("outcome")
    labeled_df  =  log.merge(outcome, on='case_nr', how='left')
    labeled_df = labeled_df.sort_values(by=["case_nr", "timestamp"])
    
    # Balance the dataset
    case_nr_outcome = labeled_df.groupby('case_nr', as_index=False)['outcome'].last()
    rus  = RandomUnderSampler(random_state=42)
    case_nr_res, _ = rus.fit_resample(case_nr_outcome['case_nr'].values.reshape(-1, 1),  # type: ignore
                                      case_nr_outcome['outcome'].values)
    balanced_df = labeled_df[labeled_df['case_nr'].isin(case_nr_res.squeeze())]

    # Reduce number of features
    included_features =  ["case_nr", "activity", "amount", "unc_quality", "est_quality", "timestamp", 
                          "interest_rate", "discount_factor", "elapsed_time", "outcome"]
    reduced_df = balanced_df.loc[:,  included_features]
    reduced_df = reduced_df.sort_values(by=["case_nr"])

    grouped = reduced_df.groupby("case_nr", as_index=False)["timestamp"].agg(
        ["min", "max"]
    )

    # Splitting into train and test
    ### TEST SET ###
    test_len = 0.2
    first_test_case_nr = int(len(grouped) * (1 - test_len))
    first_test_start_time = (
        grouped["min"].sort_values().values[first_test_case_nr]
    )
    # retain cases that end after first_test_start time
    test_case_nrs = grouped.loc[
        grouped["max"].values >= first_test_start_time, "case_nr"
    ]
    test = reduced_df[reduced_df["case_nr"] \
                      .isin(test_case_nrs)] \
                      .sort_values(by=['case_nr', 'timestamp']) \
                      .reset_index(drop=True)

    #### TRAINING SET ###
    train = reduced_df[~reduced_df["case_nr"] \
                       .isin(test_case_nrs)] \
                       .sort_values(by=['case_nr', 'timestamp']) \
                       .reset_index(drop=True)
    
    train = train.drop(columns=["timestamp"], axis=1)
    test = test.drop(columns=["timestamp"], axis=1)

    # Filling in missing values
    train['interest_rate'] = train['interest_rate'].fillna(0.0)
    test['interest_rate'] = test['interest_rate'].fillna(0.0)
    train['discount_factor'] = train['discount_factor'].fillna(0.0)
    test['discount_factor'] = test['discount_factor'].fillna(0.0)

    # Scaling numeric features
    sc = StandardScaler()
    train.loc[:, NUMERIC_FEATURES] = sc.fit_transform(train[NUMERIC_FEATURES])
    test.loc[:, NUMERIC_FEATURES] = sc.transform(test[NUMERIC_FEATURES])

    event_features = EventFeatures(
        categorical=training_config["categorical_features"],
        numerical=training_config["continuous_features"],
    )

    event_targets = EventTargets(
        categorical=training_config["categorical_targets"],
        numerical=training_config["continuous_targets"],
    )

    train_log = EventLog(
        dataframe=train,
        case_id="case_nr",
        features=event_features,
        targets=event_targets,
        train_split=True,
        name=training_config["log"],
    )

    test_log = EventLog(
        dataframe=test,
        case_id="case_nr",
        features=event_features,
        targets=event_targets,
        train_split=False,
        name=training_config["log"],
        vocabs=train_log.get_vocabs(),
    )

    dataset_device = (
        training_config["device"]
        if training_config["backbone"]
        not in ["gpt2", "llama32-1b", "llama2-7b", "qwen25-05b"]
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
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    model_config = get_model_config(train_log, training_config)
    model_config.pop("categorical_targets", None)
    model_config.pop("numerical_targets", None)
    model = OutcomePredictor(**model_config).to(device=training_config["device"])

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    training_config.update(
        {
            "total_params": all_param,
            "trainable_params": trainable_params,
        }
    )

    use_wandb = training_config.pop("wandb")
    persist_model = training_config.pop("persist_model")
    if use_wandb and WANDB_AVAILABLE:
        if (
            "freeze_layers" in training_config
            and training_config["freeze_layers"] is not None
        ):
            training_config["freeze_layers"] = ",".join(
                [str(i) for i in training_config["freeze_layers"]]
            )
        wandb.init(project=training_config.pop("project_name"), config=training_config)
        wandb.watch(model, log="all")

    print("=" * 80)
    print("Training")
    train_engine(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        config=training_config,
        use_wandb=use_wandb,
        persist_model=persist_model,
    )
    print("=" * 80)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    config_path = "configs/train_lstm_args_for_op_synth_data.txt"
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
        "continuous_features": (
            NUMERIC_FEATURES
            if (
                args.continuous_features is not None
                and "all" in args.continuous_features
            )
            else args.continuous_features
        ),
        "categorical_targets": args.categorical_targets,
        "continuous_targets": args.continuous_targets,
        "strategy": args.strategy,
        "pos_encoding_form": args.pos_encoding_form,
        "pos_encoding_strategy": args.pos_encoding_strategy,
    }
    # if is_duplicate(training_config):
    #     print("Duplicate configuration. Skipping...")
    #     exit(0)

    pprint.pprint(training_config)
    print("=" * 80)
    main(training_config)
