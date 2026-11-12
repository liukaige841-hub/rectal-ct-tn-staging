"""安全的权重读取与分类阈值工具。"""
from pathlib import Path

import numpy as np
import torch


def load_state_dict(path, device: torch.device, preferred_keys=("state_dict", "model")) -> dict:
    weight_path = Path(path).expanduser()
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"权重文件不存在：{weight_path}。公开包不含训练权重，必须分别提供匹配的分割、T、N 权重。"
        )
    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"权重格式无效：{weight_path}")
    state = None
    for key in preferred_keys:
        if key in checkpoint:
            state = checkpoint[key]
            break
    if state is None and checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        state = checkpoint
    if not isinstance(state, dict) or not state:
        raise ValueError(f"权重中未找到非空 state_dict：{weight_path}")
    return state


def positive_probability(logits: torch.Tensor) -> float:
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError(f"二分类 logits 应为 [1,2]，实际为 {tuple(logits.shape)}")
    return float(torch.softmax(logits.float(), dim=1)[0, 1].item())


def binary_prediction(probability: float, threshold: float = 0.5) -> int:
    """T/N 分类采用大于等于阈值；分割阈值逻辑另行定义。"""
    if not 0.0 <= probability <= 1.0 or not np.isfinite(probability):
        raise ValueError("概率必须是 [0,1] 内的有限数")
    return int(probability >= threshold)
