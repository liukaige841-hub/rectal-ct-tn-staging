"""最终 2D U-Net 分割训练入口。"""
import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.segmentation import UNet, bce_dice_loss, dice_score, weighted_bce_dice


class SegmentationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, csv_parent: Path, image_size: int, training: bool):
        self.frame = frame.reset_index(drop=True)
        self.csv_parent = csv_parent
        self.image_size = image_size
        self.training = training
        self.weights = (self.frame["sample_w"].to_numpy(dtype=np.float32)
                        if "sample_w" in self.frame else np.ones(len(self.frame), dtype=np.float32))

    def __len__(self):
        return len(self.frame)

    def _path(self, value):
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else self.csv_parent / path

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self._path(row["image_path"])) as source:
            image = source.convert("L").resize((self.image_size, self.image_size),
                                               Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
        with Image.open(self._path(row["mask_path"])) as source:
            mask = source.convert("L").resize((self.image_size, self.image_size),
                                              Image.Resampling.NEAREST)
            mask_array = (np.asarray(mask) > 127).astype(np.float32)
        if self.training:
            if random.random() < 0.5:
                image_array = np.ascontiguousarray(image_array[:, ::-1])
                mask_array = np.ascontiguousarray(mask_array[:, ::-1])
            if random.random() < 0.5:
                image_array = np.ascontiguousarray(image_array[::-1, :])
                mask_array = np.ascontiguousarray(mask_array[::-1, :])
            turns = random.randint(0, 3)
            if turns:
                image_array = np.ascontiguousarray(np.rot90(image_array, turns))
                mask_array = np.ascontiguousarray(np.rot90(mask_array, turns))
            gamma = random.uniform(0.8, 1.25)
            image_array = np.power(np.clip(image_array, 0, 1), gamma)
            image_array = np.clip(image_array + random.uniform(-0.08, 0.08), 0, 1)
        return (torch.from_numpy(image_array[None]), torch.from_numpy(mask_array[None]),
                torch.tensor(float(self.weights[index])))


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    all_dice, positive_dice, losses = [], [], []
    for image, mask, _ in loader:
        image, mask = image.to(device), mask.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(image)
            loss = bce_dice_loss(logits, mask)
        dice = dice_score((torch.sigmoid(logits.float()) > 0.5).float(), mask)
        values = dice.cpu().tolist()
        positive = (mask.sum(dim=(1, 2, 3)) > 0).cpu().tolist()
        all_dice.extend(values)
        positive_dice.extend(value for value, flag in zip(values, positive) if flag)
        losses.append(float(loss))
    return (float(np.mean(all_dice)), float(np.mean(losses)),
            float(np.mean(positive_dice)) if positive_dice else 0.0)


def train(config: dict):
    training = config["train"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    index_csv = Path(training["index_csv"]).expanduser().resolve()
    frame = pd.read_csv(index_csv)
    required = {"image_path", "mask_path", "has_mask", "split"}
    if required - set(frame.columns):
        raise ValueError(f"分割索引缺少列：{sorted(required - set(frame.columns))}")
    train_frame = frame[frame["split"].astype(str) == "train"].copy()
    val_frame = frame[frame["split"].astype(str) == "val"].copy()
    if train_frame.empty or val_frame.empty:
        raise ValueError("分割索引必须包含非空 train 和 val")
    fraction = float(training["neg_frac"])
    if fraction < 1.0:
        positive = train_frame[train_frame["has_mask"].astype(int) == 1]
        negative = train_frame[train_frame["has_mask"].astype(int) == 0].sample(
            frac=fraction, random_state=seed)
        train_frame = pd.concat([positive, negative])
    if training.get("hard_negative_csv"):
        hard_path = Path(training["hard_negative_csv"]).expanduser().resolve()
        hard = pd.read_csv(hard_path)
        hard_images = set(hard["image_path"].astype(str))
        train_frame["sample_w"] = np.where(train_frame["image_path"].astype(str).isin(hard_images),
                                           float(training["hard_neg_weight"]), 1.0)
    image_size = int(config["data"]["image_size"])
    train_dataset = SegmentationDataset(train_frame, index_csv.parent, image_size, True)
    val_dataset = SegmentationDataset(val_frame, index_csv.parent, image_size, False)
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(training.get("amp", True)) and device.type == "cuda"
    model = UNet(base=int(config["model"]["base"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output = Path(training["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    stale = 0
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_dice_pos"])
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            losses = []
            for image, mask, weights in train_loader:
                image, mask, weights = image.to(device), mask.to(device), weights.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(image)
                    loss = weighted_bce_dice(logits, mask, weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
            val_dice, val_loss, val_positive = evaluate(model, val_loader, device, use_amp)
            train_loss = float(np.mean(losses)) if losses else float("nan")
            writer.writerow([epoch, train_loss, val_loss, val_dice, val_positive])
            handle.flush()
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                  f"val_dice={val_dice:.6f} val_dice_pos={val_positive:.6f}", flush=True)
            checkpoint = {"model": model.state_dict(), "config": copy.deepcopy(config),
                          "epoch": epoch, "base": int(config["model"]["base"]),
                          "img_size": image_size}
            torch.save(checkpoint, output / "last.pt")
            if val_positive > best:
                best = val_positive
                stale = 0
                checkpoint["val_dice_pos"] = val_positive
                torch.save(checkpoint, output / "best.pt")
            else:
                stale += 1
                if stale >= int(training["patience"]):
                    print(f"早停：val_dice_pos 连续 {stale} 轮未提升", flush=True)
                    break
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--index-csv")
    parser.add_argument("--hard-negative-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {"index_csv": args.index_csv, "hard_negative_csv": args.hard_negative_csv,
                 "output_dir": args.output_dir, "epochs": args.epochs, "batch_size": args.batch_size}
    config["train"].update({key: value for key, value in overrides.items() if value is not None})
    best = train(config)
    print(f"最佳分割权重：{best}")


if __name__ == "__main__":
    main()
