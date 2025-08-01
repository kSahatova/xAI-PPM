import torch
import numpy as np
import pandas as pd
from typing import Tuple
import shap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from skpm import event_logs
from skpm.event_logs.split import unbiased
from skpm.feature_extraction import TimestampExtractor

from ppm.datasets import ContinuousTraces
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets
from ppm.datasets.utils import continuous
from ppm.models import NextEventPredictor
from ppm.engine.utils import load_checkpoint

from utils import parse_args, get_model_config
from cohortshapley.cohortshapley_el import CohortExplainer

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


def prepare_data(
    df: pd.DataFrame, unbiased_split_params: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    train = train.drop(columns=["time:timestamp"])
    test = test.drop(columns=["time:timestamp"])

    train = train.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )
    test = test.rename(
        columns={"case:concept:name": "case_id", "concept:name": "activity"}
    )

    sc = StandardScaler()
    columns = NUMERICAL_FEATURES + ["remaining_time"]
    # columns = ["accumulated_time", "remaining_time"]
    train.loc[:, columns] = sc.fit_transform(train[columns])
    test.loc[:, columns] = sc.transform(test[columns])

    return train, test


def get_model_prediction(model: torch.nn.Module,
                         batch: Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor], 
                         device: str):
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        x_cat, x_num, _, _ = batch
        x_cat, x_num = (
            x_cat.to(device),
            x_num.to(device),
        )

        attention_mask = (x_cat[..., 0] != 0).long()
        out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
        return out['next_activity'][0][-1].detach().cpu().numpy()


def plot_bar_plot(shapley_values, feature_names, max_display=10):
    """
    Create a horizontal bar plot similar to SHAP's bar plot
    """
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(shapley_values))[::-1][:max_display]
    sorted_values = shapley_values[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors based on positive/negative
    colors = ['red' if val < 0 else 'blue' for val in sorted_values]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_values))
    bars = ax.barh(y_pos, sorted_values, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_values)):
        ax.text(val + (0.01 if val >= 0 else 0.06), i, f'{val:.4f}', 
                va='center', ha='left' if val >= 0 else 'right', fontweight='bold',
                fontsize=9, color='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('SHAP Value (impact on model output)')
    ax.set_title('SHAP Summary Plot - Bar')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/bar_plot_cohort_shapley_0.png', dpi=300)
    return fig, ax


def model_predict_proba(model, batch, device):
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        x_cat, x_num, _, _ = batch
        x_cat, x_num = (
            x_cat.to(device),
            x_num.to(device),
        )

        attention_mask = (x_cat[..., 0] != 0).long()
        out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)
        probs = F.softmax(out['next_activity'][0][-1], dim=-1)
    return probs



def main(config: dict):
    log = getattr(event_logs, config["log"])()
    train, test = prepare_data(
        log.dataframe, log.unbiased_split_params
    )  # this is my current code for the fine-tuning experiments
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

    test_dataset = ContinuousTraces(
        log=test_log,
        refresh_cache=True,
        device=dataset_device,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=100, #config["batch_size"],
        shuffle=False,
        collate_fn=continuous,
    )

    # Loading a pre-trained model
    model_config = get_model_config(train_log, config)

    model = NextEventPredictor(**model_config).to(device=config["device"])
    ckpt = load_checkpoint(config["checkpoint_path"], map_location=config["device"])
    model.load_state_dict(ckpt)#['net'])

    batch = next(iter(test_loader))
    pred_probs = model_predict_proba(model, batch, dataset_device)


    # Instantiate CohortExplainer
    subject_id = 0
    y = []
    for batch in test_loader:
        y.append(get_model_prediction(model, batch, config["device"]))
    # explainer = CohortExplainer(model, test_dataset, y=np.asarray(y))
    # shapley_val, shapley_val2 = explainer.compute_cohort_shapley(subject_id, features_num=11)
    

if __name__ == "__main__":
    args = parse_args()

    config = {
        # args to pop before logging
        "project_name": args.project_name,
        "checkpoint_path": args.checkpoint_path,
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
    }

   
    main(config)

