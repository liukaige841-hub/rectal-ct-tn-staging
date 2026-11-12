"""DICOM/NIfTI 到 T/N 概率与类别的完整推理流程。"""
import json
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.imaging import load_volume
from src.inference_utils import binary_prediction, positive_probability
from src.models import build_classifier, load_classifier_weights
from src.postprocess import postprocess_mask
from src.roi import build_25d_roi, normalize_roi, save_roi_pngs
from src.segmentation import UNet, abdominal_window, load_segmentation_weights, segment_volume


def _resolve_reference(package_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else package_root / path


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境不可用")
    return device


@torch.no_grad()
def predict(input_path, segmentation_weights, t_weights, n_weights, config_path,
            save_roi_dir=None) -> dict:
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    package_root = config_path.parent.parent
    # 在读取输入影像之前加载全部三个权重；任何缺失或不匹配均不会生成分期结果。
    device = _device(str(config.get("runtime", {}).get("device", "auto")))
    segmentation_config = config["segmentation"]
    segmentation_model = UNet(base=int(segmentation_config["base"])).to(device)
    load_segmentation_weights(segmentation_model, segmentation_weights, device)
    t_config = load_config(_resolve_reference(package_root, config["classification"]["T_config"]))
    n_config = load_config(_resolve_reference(package_root, config["classification"]["N_config"]))
    t_model = build_classifier(t_config, pretrained=False).to(device)
    n_model = build_classifier(n_config, pretrained=False).to(device)
    load_classifier_weights(t_model, t_weights, device)
    load_classifier_weights(n_model, n_weights, device)

    volume, _ = load_volume(input_path)
    gray = abdominal_window(volume, float(segmentation_config["window_center"]),
                            float(segmentation_config["window_width"]))
    raw_mask = segment_volume(segmentation_model, gray, device,
                              image_size=int(segmentation_config["image_size"]),
                              threshold=float(segmentation_config["threshold"]),
                              batch_size=int(segmentation_config["batch_size"]))
    mask, postprocess_info = postprocess_mask(raw_mask, **config["postprocess"])
    roi, roi_info = build_25d_roi(gray, mask, **config["roi"], training=False)
    normalized = normalize_roi(roi)
    batch = torch.from_numpy(normalized[None]).to(device=device, dtype=torch.float32)
    t_probability = positive_probability(t_model(batch))
    n_probability = positive_probability(n_model(batch))
    t_threshold = float(config["classification"]["T_threshold"])
    n_threshold = float(config["classification"]["N_threshold"])
    result = {
        "T": {
            "model": "T-BIN-AUTO",
            "positive_class": "T>=3",
            "probability": t_probability,
            "threshold": t_threshold,
            "predicted_class": binary_prediction(t_probability, t_threshold),
            "predicted_label": "T>=3" if binary_prediction(t_probability, t_threshold) else "T<=2",
        },
        "N": {
            "model": "N-BIN-AUTO",
            "positive_class": "N+",
            "probability": n_probability,
            "threshold": n_threshold,
            "predicted_class": binary_prediction(n_probability, n_threshold),
            "predicted_label": "N+" if binary_prediction(n_probability, n_threshold) else "N0",
        },
        "segmentation": {
            "model": "SEG-2D-UNET",
            "threshold": float(segmentation_config["threshold"]),
            "raw_mask_voxels": int(raw_mask.sum()),
            "postprocessed_mask_voxels": int(mask.sum()),
            **postprocess_info,
        },
        "roi": roi_info,
    }
    if save_roi_dir is not None:
        save_roi_pngs(roi, Path(save_roi_dir) / "roi")
    return result


def write_result(result: dict, output_path) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def result_json(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
