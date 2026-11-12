"""关键层选择、二维边界框与三通道 2.5D ROI。"""
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def select_key_slice(mask: np.ndarray, key_z_gt: int | None = None, training: bool = False) -> tuple[int, bool]:
    """返回关键层及是否使用空掩膜兜底。训练兜底必须提供人工关键层。"""
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 3:
        raise ValueError(f"掩膜必须为三维，实际为 {binary.shape}")
    if binary.any():
        return int(np.argmax(binary.sum(axis=(0, 1)))), False
    if training:
        if key_z_gt is None:
            raise ValueError("训练侧空掩膜兜底要求提供人工 GT 关键层 key_z")
        return int(np.clip(key_z_gt, 0, binary.shape[2] - 1)), True
    return binary.shape[2] // 2, True


def expanded_union_bbox(
    mask: np.ndarray, key_z: int, bbox_window: int = 8, margin_ratio: float = 0.4
) -> tuple[int, int, int, int]:
    """在 rot90 显示坐标中计算 key±window 掩膜并集及双向 40% 外扩。"""
    n_slices = mask.shape[2]
    lower = max(0, key_z - bbox_window)
    upper = min(n_slices - 1, key_z + bbox_window)
    union = np.zeros((mask.shape[1], mask.shape[0]), dtype=bool)
    for z in range(lower, upper + 1):
        union |= np.rot90(mask[:, :, z])
    if not union.any():
        center = union.shape[0] // 2
        half = int(union.shape[0] * 0.25)
        return center - half, center - half, center + half, center + half
    ys, xs = np.nonzero(union)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    margin_x = (x1 - x0) * margin_ratio
    margin_y = (y1 - y0) * margin_ratio
    cx0 = int(max(0, x0 - margin_x))
    cy0 = int(max(0, y0 - margin_y))
    cx1 = int(min(union.shape[1], x1 + margin_x))
    cy1 = int(min(union.shape[0], y1 + margin_y))
    return cx0, cy0, cx1, cy1


def build_25d_roi(
    gray: np.ndarray,
    mask: np.ndarray,
    image_size: int = 224,
    neighbor_slices: int = 1,
    bbox_window: int = 8,
    margin_ratio: float = 0.4,
    key_z_gt: int | None = None,
    training: bool = False,
) -> tuple[np.ndarray, dict]:
    """返回 uint8 [3,H,W]，通道顺序为 z-1、z、z+1，边界复制。"""
    gray = np.asarray(gray)
    mask = np.asarray(mask).astype(bool)
    if gray.shape != mask.shape or gray.ndim != 3:
        raise ValueError(f"灰度与掩膜形状必须一致且为三维：{gray.shape} vs {mask.shape}")
    if neighbor_slices != 1:
        raise ValueError("最终模型固定使用 neighbor_slices=1")
    key_z, fallback = select_key_slice(mask, key_z_gt=key_z_gt, training=training)
    bbox = expanded_union_bbox(mask, key_z, bbox_window=bbox_window, margin_ratio=margin_ratio)
    cx0, cy0, cx1, cy1 = bbox
    channels = []
    for z in (key_z - 1, key_z, key_z + 1):
        z_clamped = min(max(z, 0), gray.shape[2] - 1)
        display = np.rot90(gray[:, :, z_clamped])
        crop = display[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            raise ValueError("ROI 裁剪结果为空，请检查掩膜和输入几何")
        resized = Image.fromarray(crop.astype(np.uint8), mode="L").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        channels.append(np.asarray(resized, dtype=np.uint8))
    info = {"key_z": key_z, "empty_mask_fallback": fallback, "bbox_xyxy": list(bbox)}
    return np.stack(channels), info


def normalize_roi(roi: np.ndarray) -> np.ndarray:
    values = np.asarray(roi, dtype=np.float32) / 255.0
    if values.shape[0] != 3:
        raise ValueError("2.5D ROI 必须为三个通道")
    return (values - IMAGENET_MEAN) / IMAGENET_STD


def save_roi_pngs(roi: np.ndarray, output_stem: str | Path) -> list[Path]:
    """按 <stem>_zm1.png、<stem>.png、<stem>_zp1.png 保存。"""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_name(stem.name + "_zm1.png"), stem.with_suffix(".png"),
             stem.with_name(stem.name + "_zp1.png")]
    for channel, path in zip(roi, paths):
        Image.fromarray(channel, mode="L").save(path)
    return paths
