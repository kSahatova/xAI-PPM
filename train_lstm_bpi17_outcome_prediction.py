import pprint
import wandb
# from wandb import WANDB_AVAILABLE

import torch
from torch.utils.data import DataLoader

from ppm.datasets.utils import continuous
from ppm.datasets import ContinuousTraces, DatasetSchemas
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets

from skpm.event_logs import BPI17

from ppm.engine.op import train_engine
from ppm.models import OutcomePredictor

from ppm.utils import parse_args, add_outcome_labels, prepare_data, get_model_config


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

PRETRAINED_CONFIGS = {
    "gpt2": {
        "name": "openai-community/gpt2",
        "embedding_size": 768,
        "hidden_size": 768,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "pm-gpt2": {
        "name": "models/pm-gpt2",
        "embedding_size": 768,
        "hidden_size": 768,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "llama32-1b": {
        "name": "meta-llama/Llama-3.2-1B",
        "embedding_size": 2048,
        "hidden_size": 2048,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "embedding_size": 4096,
        "hidden_size": 4096,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    "qwen25-05b": {
        "name": "Qwen/Qwen2.5-0.5B",
        "embedding_size": 896,
        "hidden_size": 896,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
}

EVENT_LOGS = {"BPI17": BPI17}


def main(training_config: dict):
    log = EVENT_LOGS[training_config["log"]]()

    labels_dict = {"O_Cancelled": 0, "O_Accepted": 1, "O_Refused": 2}
    column_schema = getattr(DatasetSchemas, training_config["log"])()
    labeled_df = add_outcome_labels(log.dataframe, column_schema, labels_dict)

    # Remove O_Refused to convert the task to a binary classification
    binary_labeled_df = labeled_df[labeled_df["outcome"] != 2]
    print(
        "Outcomes of the cases are ",
        binary_labeled_df["last_o_activity"].unique().tolist(),
    )
    result = prepare_data(
        binary_labeled_df,
        log.unbiased_split_params,
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
        case_id="case_id",
        features=event_features,
        targets=event_targets,
        train_split=True,
        name=training_config["log"],
    )

    test_log = EventLog(
        dataframe=test,
        case_id="case_id",
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

    model_config = get_model_config(train_log, training_config, PRETRAINED_CONFIGS)
    # TODO: adjust to the logic of next activity predcition (!remove this fast fix)
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
    config_path = "configs/train_lstm_args_for_op.txt"
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
            NUMERICAL_FEATURES
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


    pprint.pprint(training_config)
    print("=" * 80)
    main(training_config)
