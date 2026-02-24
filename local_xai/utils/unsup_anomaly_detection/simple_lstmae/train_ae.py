import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import DataLoader

from skpm import event_logs

from ppm.datasets import ContinuousTraces
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets
from ppm.datasets.utils import continuous
from ppm.utils import parse_args, prepare_data

from ae_model import LSTMAutoencoder
from ae_trainer import LSTMAETrainer

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

def main(config: dict):
    log = getattr(event_logs, config["log"])()

    result = prepare_data(
            log.dataframe,
            log.unbiased_split_params,
            NUMERICAL_FEATURES,
        )
    # prepare_data may return either (train, test) or (train, test, train_timestamps, test_timestamps)
    if isinstance(result, tuple) and len(result) == 2:
        train, test = result
    else:
        raise ValueError("Unexpected return value from prepare_data()")

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
        if config["backbone"]
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

    # Initialize model
    model = LSTMAutoencoder(
        input_dim=11,
        hidden_dim=config["hidden_dim"],
        latent_dim=config["lattent_dim"],
        num_layers=config["n_layers"],
        dropout=0.2
    )

    # Initialize trainer
    trainer = LSTMAETrainer(model)

    # Train the model
    print("Training LSTM Autoencoder...")
    trainer.train(train_loader, epochs=config["epochs"], lr=config["lr"], patience=10)

    # TODO: check that  checkpoints saving is correct
    checkpoints_path = "output/checkpoints/lstmae_bpi17.pth"
    torch.save(model.state_dict(), checkpoints_path)
    print(f"Model checkpoints saved at: {checkpoints_path}")

    # Plot training history
    trainer.plot_training_history()


if __name__ == "__main__":
    config_path = r'D:\PycharmProjects\xAI-PPM\configs\train_lstmae_args.txt'

    args = parse_args(config_path=config_path)
    args.dataset = 'BPI17'

    config = {
        # args to pop before logging
        "project_name": args.project_name,
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
        "batch_size": args.batch_size, 
        "epochs": args.epochs,
        "lr": args.lr,
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
        "hidden_dim": 64,
        "lattent_dim": 32
    }

    main(config)