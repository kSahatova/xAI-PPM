import argparse
import torch
from torch.utils.data import DataLoader
from ppm.datasets.event_logs import EventLog


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
