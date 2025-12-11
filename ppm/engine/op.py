import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm

from ppm.engine.utils import save_checkpoint
from ppm.metrics import MetricsTracker

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_step(
    model,
    data_loader,
    optimizer,
    tracker: MetricsTracker,
    device="cuda",
    scaler=None,
    grad_clip=None,
):
    model.train()
    metrics = {
        target: {metric: 0.0 for metric in tracker.metrics[target]}
        for target in tracker.metrics
        if target.startswith("train")
    }
    total_targets=0
    valid_positions = {}
    for batch, items in enumerate(data_loader):
        x_cat, x_num, y_cat, y_num = items
        x_cat, x_num, y_cat, y_num = (
            x_cat.to(device),
            x_num.to(device),
            y_cat.to(device),
            y_num.to(device),
        )

        attention_mask = (x_cat[..., 0] != 0).long()
        total_targets += attention_mask.sum() #y_cat.shape[0]  

        optimizer.zero_grad()
        out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask) # sigmoid inside model

        # to consider: weighted loss over time to bias the model
        # toward early predictions
        loss = F.binary_cross_entropy(
            out.squeeze(-1)[attention_mask.bool()],
            y_cat.squeeze(-1)[attention_mask.bool()].to(torch.float),
            reduction="sum",
        )
        predictions = ((out.squeeze(-1)) > 0.5).float()
        acc = (predictions.squeeze(-1)[attention_mask.bool()] == y_cat.squeeze(-1)[attention_mask.bool()]).sum().item()

        metrics["train_outcome"]["loss"] += loss.item()
        metrics["train_outcome"]["acc"] += acc
        
        max_len = attention_mask.size(1)
        idxs = torch.arange(max_len).unsqueeze(0).to(device)  # [1, S]
        lengths = attention_mask.sum(dim=-1)  # [B]
        mask = idxs < lengths.unsqueeze(1)  # [B, S]
        
        correct = (predictions.squeeze() == y_cat.squeeze()) & mask
        for t in range(max_len):
            valid = mask[:, t]
            if valid.any():
                acc_t = correct[:, t][valid].float().sum()
            else:
                acc_t = torch.tensor(float("nan"))
            metrics["train_outcome"].setdefault(f"acc_pos_{t}", 0.0)
            metrics["train_outcome"][f"acc_pos_{t}"] += acc_t.item()
            valid_positions.setdefault(f"acc_pos_{t}", 0)
            valid_positions[f"acc_pos_{t}"] += valid.sum().item()

        loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    for target in metrics:
        for k in metrics[target].keys():
            if k.startswith("acc_pos_"):
                metrics[target][k] /= valid_positions[k]
            else:
                metrics[target][k] /= total_targets

        tracker.update(target, **metrics[target])

    return tracker


def eval_step(model, data_loader, tracker: MetricsTracker, device="cuda"):
    model.eval()
    metrics = {
        target: {metric: 0.0 for metric in tracker.metrics[target]}
        for target in tracker.metrics
        if target.startswith("test")
    }
    # total_targets = len(data_loader)
    # total_targets = len(data_loader.dataset.traces)
    total_targets=0
    valid_positions = {}
    with torch.inference_mode():
        for batch, items in enumerate(data_loader):
            x_cat, x_num, y_cat, y_num = items
            x_cat, x_num, y_cat, y_num = (
                x_cat.to(device),
                x_num.to(device),
                y_cat.to(device),
                y_num.to(device),
            )

            # y_cat = y_cat[:, -1, :]
            attention_mask = (x_cat[..., 0] != 0).long()
            total_targets += attention_mask.sum().item()

            # with torch.autocast(device_type=device, dtype=torch.float16):
            out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)

            batch_loss = 0.0
            # mask = attention_mask.bool().view(-1)

            loss = F.binary_cross_entropy(
                out.squeeze(-1)[attention_mask.bool()],
                y_cat.squeeze(-1)[attention_mask.bool()].to(torch.float),
                reduction="sum",
            )
            predictions = ((out.squeeze(-1)) > 0.5).float()
            acc = (predictions.squeeze(-1)[attention_mask.bool()] == y_cat.squeeze(-1)[attention_mask.bool()]).sum().item()

            batch_loss += loss
            metrics["test_outcome"]["loss"] += loss.item()
            metrics["test_outcome"]["acc"] += acc
            
            max_len = attention_mask.size(1)
            idxs = torch.arange(max_len).unsqueeze(0).to(device)  # [1, S]
            lengths = attention_mask.sum(dim=-1)  # [B]
            mask = idxs < lengths.unsqueeze(1)  # [B, S]
            
            correct = (predictions.squeeze() == y_cat.squeeze()) & mask
            for t in range(max_len):
                valid = mask[:, t]
                if valid.any():
                    acc_t = correct[:, t][valid].float().sum()
                else:
                    acc_t = torch.tensor(float("nan"))
                metrics["test_outcome"].setdefault(f"acc_pos_{t}", 0.0)
                metrics["test_outcome"][f"acc_pos_{t}"] += acc_t.item()
                valid_positions.setdefault(f"acc_pos_{t}", 0)
                valid_positions[f"acc_pos_{t}"] += valid.sum().item()
            # metrics["test_outcome"]["acc"] += acc

    for target in metrics:
        for k in metrics[target].keys():
            if k.startswith("acc_pos_"):
                metrics[target][k] /= valid_positions[k]
            else:
                metrics[target][k] /= total_targets

        tracker.update(target, **metrics[target])

    return tracker


def train_engine(
    model: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    config: dict,
    use_wandb: bool,
    persist_model: bool,
):
    model.to(config["device"])

    categorical_target_metrics = {
        f"{split}_{target}": ["loss", "acc"]
        for split in ["train", "test"]
        for target in train_loader.dataset.log.targets.categorical
    }

    tracker = MetricsTracker({**categorical_target_metrics})

    best_loss = torch.inf
    no_improvement = 0
    for epoch in range(config["epochs"]):
        tracker = train_step(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=config["device"],
            tracker=tracker,
            grad_clip=config["grad_clip"],
        )
        tracker = eval_step(
            model=model,
            data_loader=test_loader,
            device=config["device"],
            tracker=tracker,
        )

        print(
            f"Epoch {epoch}: ",
            " | ".join(
                f"{k}: {v:.4f}" for k, v in tracker.latest().items() if "best" not in k
            ),
        )

        if WANDB_AVAILABLE and use_wandb:
            wandb.log(tracker.latest())

        loss_key = "test_outcome_loss"
        activity_loss = tracker.latest()[loss_key]
        if persist_model:
            if activity_loss < best_loss:
                cpkt = {
                    "epoch": epoch,
                    "net": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "stoi": train_loader.dataset.log.stoi,
                    "itos": train_loader.dataset.log.itos,
                }
                save_checkpoint(
                    checkpoint=cpkt,
                    experiment_id="{}_{}_{}_{}".format(config["log"], config["backbone"], 
                                                       config["categorical_targets"][0], config["log"].lower()),
                )

        if activity_loss < best_loss:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 15:
                break

        best_loss = min(best_loss, tracker.latest()[loss_key])


    # if you dont' want to use wandb:
    import pandas as pd
    def quick_convert_to_df(data):
        df = pd.DataFrame(data)
        df = df.T.reset_index().rename(columns={'index': 'acc position', 0: 'acc'})
        df = df[df['acc position'].str.startswith('acc_pos_')]
        return df
    
    train_df = quick_convert_to_df(tracker.history()['train_outcome'])
    test_df = quick_convert_to_df(tracker.history()['test_outcome'])
    df = pd.merge(train_df, test_df, on='acc position', suffixes=('_train', '_test'))
    df.to_csv("a.csv", index=False)
    # print(df)
    

    optimizer.zero_grad()
