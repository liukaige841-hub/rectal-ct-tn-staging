"""最终 T-BIN-AUTO 或 N-BIN-AUTO 分类训练入口。"""
import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.datasets import build_dataloaders
from src.models import build_classifier


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for image, label, _ in loader:
        logits = model(image.to(device))
        labels.append(label.numpy())
        probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    prediction = (y_prob >= 0.5).astype(np.int64)
    accuracy = float(np.mean(prediction == y_true))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    return accuracy, auc


def train(config: dict, pretrained: bool | None = None):
    training = config["train"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_loader, val_loader, counts = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training["amp"]) and device.type == "cuda"
    model = build_classifier(config, pretrained=pretrained).to(device)
    class_weights = None
    if bool(training["class_weight"]):
        values = np.clip(np.asarray(counts, dtype=np.float64), 1, None)
        class_weights = torch.tensor(values.sum() / (2 * values), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=float(training["label_smoothing"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]),
                                 weight_decay=float(training["weight_decay"]))
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(training["output_dir"]).expanduser().resolve()
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    stale = 0
    print(f"任务={config['model']['name']} 训练类别计数={counts} 设备={device}", flush=True)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_acc", "val_auc", "lr"])
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0
            for image, label, _ in train_loader:
                image, label = image.to(device), label.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(image)
                    loss = criterion(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach()) * image.shape[0]
                seen += image.shape[0]
            scheduler.step()
            accuracy, auc = evaluate(model, val_loader, device)
            score = auc if np.isfinite(auc) else accuracy
            train_loss = running_loss / max(seen, 1)
            writer.writerow([epoch, train_loss, accuracy, auc, scheduler.get_last_lr()[0]])
            handle.flush()
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_acc={accuracy:.6f} "
                  f"val_auc={auc:.6f} lr={scheduler.get_last_lr()[0]:.3e}", flush=True)
            state = {"state_dict": model.state_dict(), "config": copy.deepcopy(config),
                     "epoch": epoch, "val_metrics": {"acc": accuracy, "auc": auc}}
            torch.save(state, checkpoint_dir / "last.pt")
            if score > best_score:
                best_score = score
                stale = 0
                torch.save(state, checkpoint_dir / "best.pt")
            else:
                stale += 1
                if (stale >= int(training["early_stop_patience"])
                        and epoch >= int(training["min_epochs"])):
                    print(f"早停：连续 {stale} 轮未提升", flush=True)
                    break
    return checkpoint_dir / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/T_staging.yaml 或 configs/N_staging.yaml")
    parser.add_argument("--labels-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="仅用于离线结构调试；最终训练配置为 pretrained=true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.labels_csv:
        config["data"]["labels_csv"] = args.labels_csv
    overrides = {"output_dir": args.output_dir, "epochs": args.epochs, "batch_size": args.batch_size}
    config["train"].update({key: value for key, value in overrides.items() if value is not None})
    best = train(config, pretrained=False if args.no_pretrained else None)
    print(f"最佳分类权重：{best}")


if __name__ == "__main__":
    main()
