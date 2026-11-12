"""最终 2D U-Net 及逐层分割预处理/推理。"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """输入和输出均为 [B,1,H,W]；输出为 logits。"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        b = base
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.bott = ConvBlock(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def abdominal_window(volume: np.ndarray, center: float = 50.0, width: float = 350.0) -> np.ndarray:
    """HU 体数据窗化到 uint8；默认范围 [-125, 225]。"""
    values = np.asarray(volume, dtype=np.float32)
    lower = center - width / 2.0
    upper = center + width / 2.0
    return (np.clip((values - lower) / (upper - lower), 0.0, 1.0) * 255.0).astype(np.uint8)


def prepare_segmentation_slice(gray: np.ndarray, z: int, image_size: int = 256) -> np.ndarray:
    """逐层 rot90、双线性缩放并归一化到 [0,1]。"""
    slice_2d = np.rot90(gray[:, :, z])
    resized = Image.fromarray(slice_2d, mode="L").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def probability_to_mask(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """分割使用严格大于阈值；等于阈值时判为背景。"""
    return np.asarray(probability) > threshold


@torch.no_grad()
def segment_volume(
    model: nn.Module,
    gray: np.ndarray,
    device: torch.device,
    image_size: int = 256,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> np.ndarray:
    """返回原网格布尔掩膜，形状与输入 [X,Y,Z] 一致。"""
    if gray.ndim != 3:
        raise ValueError(f"灰度体数据必须为三维，实际为 {gray.shape}")
    model.eval()
    original_x, original_y, n_slices = gray.shape
    output = np.zeros(gray.shape, dtype=bool)
    for start in range(0, n_slices, batch_size):
        stop = min(start + batch_size, n_slices)
        slices = [prepare_segmentation_slice(gray, z, image_size) for z in range(start, stop)]
        batch = torch.from_numpy(np.stack(slices)[:, None]).to(device=device, dtype=torch.float32)
        probabilities = torch.sigmoid(model(batch).float())[:, 0].cpu().numpy()
        binary = probability_to_mask(probabilities, threshold)
        for offset, z in enumerate(range(start, stop)):
            restored = Image.fromarray(binary[offset].astype(np.uint8) * 255, mode="L").resize(
                (original_x, original_y), Image.Resampling.NEAREST
            )
            output[:, :, z] = np.rot90(np.asarray(restored) > 0, -1)
    return output


def weighted_bce_dice(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """逐样本加权 BCE + soft Dice。"""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=(1, 2, 3))
    probabilities = torch.sigmoid(logits)
    numerator = 2.0 * (probabilities * targets).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps
    per_sample = bce + 1.0 - numerator / denominator
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weights = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
    return weighted_bce_dice(logits, targets, weights, eps)


def dice_score(prediction: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    numerator = 2.0 * (prediction * targets).sum(dim=(2, 3))
    denominator = prediction.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return torch.where(
        denominator > 0,
        (numerator + eps) / (denominator + eps),
        torch.ones_like(denominator),
    )


def load_segmentation_weights(model: nn.Module, path: str | Path, device: torch.device) -> None:
    from src.inference_utils import load_state_dict

    model.load_state_dict(load_state_dict(path, device, preferred_keys=("model", "state_dict")), strict=True)
