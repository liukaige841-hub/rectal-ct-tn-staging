"""最终三通道 2.5D 分类数据集。"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.roi import IMAGENET_MEAN, IMAGENET_STD


def _resolve(base: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path


class ROIDataset(Dataset):
    """清单列：sample_id,image_path,label,split；路径相对于清单所在目录。"""

    def __init__(self, config: dict, split: str, training: bool, labels_csv: str | None = None):
        data = config["data"]
        self.csv_path = Path(labels_csv or data["labels_csv"]).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"分类标签清单不存在：{self.csv_path}")
        frame = pd.read_csv(self.csv_path, dtype={data.get("sample_id_col", "sample_id"): str})
        required = {data["image_col"], data["label_col"], "split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"分类清单缺少列：{sorted(missing)}")
        self.frame = frame[frame["split"].astype(str) == split].reset_index(drop=True)
        self.frame = self.frame[self.frame[data["label_col"]].notna()].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"split={split} 的有效分类样本为 0")
        self.labels = self.frame[data["label_col"]].astype(int).to_numpy(dtype=np.int64)
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("最终 T/N 任务标签必须为 0 或 1")
        self.image_col = data["image_col"]
        self.id_col = data.get("sample_id_col", "sample_id")
        self.image_size = int(data["image_size"])
        self.training = training
        if int(data.get("neighbor_slices", 1)) != 1:
            raise ValueError("最终模型固定使用 neighbor_slices=1")

    def __len__(self):
        return len(self.frame)

    def _load(self, path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"ROI 图像不存在：{path}")
        with Image.open(path) as image:
            resized = image.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            return np.asarray(resized, dtype=np.float32) / 255.0

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        center_path = _resolve(self.csv_path.parent, row[self.image_col])
        center = self._load(center_path)
        stem = center_path.with_suffix("")
        previous = center_path.with_name(stem.name + "_zm1.png")
        following = center_path.with_name(stem.name + "_zp1.png")
        channels = [self._load(previous) if previous.is_file() else center,
                    center,
                    self._load(following) if following.is_file() else center]
        tensor = torch.from_numpy(np.stack(channels))
        if self.training:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[1])
            if random.random() < 0.75:
                tensor = TF.affine(tensor, angle=random.uniform(-15, 15), translate=[0, 0],
                                   scale=random.uniform(0.9, 1.1), shear=[0, 0])
            alpha = 1.0 + random.uniform(-0.1, 0.1)
            beta = random.uniform(-0.1, 0.1)
            tensor = (tensor * alpha + beta).clamp(0.0, 1.0)
        mean = torch.from_numpy(IMAGENET_MEAN)
        std = torch.from_numpy(IMAGENET_STD)
        tensor = (tensor - mean) / std
        sample_id = str(row[self.id_col]) if self.id_col in self.frame.columns else str(index)
        return tensor, torch.tensor(int(self.labels[index]), dtype=torch.long), sample_id


def build_dataloaders(config: dict, labels_csv: str | None = None):
    training = config["train"]
    train_dataset = ROIDataset(config, "train", True, labels_csv)
    val_dataset = ROIDataset(config, "val", False, labels_csv)
    counts = np.bincount(train_dataset.labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"训练集必须同时包含两个类别，当前计数：{counts.tolist()}")
    shuffle = True
    sampler = None
    if bool(training.get("balanced_sampler", True)):
        sample_weights = 1.0 / counts[train_dataset.labels].astype(np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(train_dataset), replacement=True)
        shuffle = False
    generator = torch.Generator().manual_seed(int(training["seed"]))
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
    return train_loader, val_loader, counts.tolist()
