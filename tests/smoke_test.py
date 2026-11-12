"""无需真实数据或权重的随机张量/数组冒烟测试。"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.inference_utils import binary_prediction, load_state_dict, positive_probability
from src.models import build_classifier
from src.postprocess import postprocess_mask, z_runs
from src.roi import build_25d_roi, normalize_roi, select_key_slice
from src.segmentation import (UNet, abdominal_window, prepare_segmentation_slice,
                              probability_to_mask, segment_volume)


def test_segmentation_forward_and_threshold():
    model = UNet(base=4).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 1, 64, 64))
    assert output.shape == (2, 1, 64, 64)
    assert not probability_to_mask(np.asarray([0.5]), 0.5).item()
    assert probability_to_mask(np.asarray([0.500001]), 0.5).item()
    gray = np.zeros((20, 24, 3), dtype=np.uint8)
    assert prepare_segmentation_slice(gray, 0, 32).shape == (32, 32)
    zero_model = UNet(base=2).eval()
    with torch.no_grad():
        for parameter in zero_model.parameters():
            parameter.zero_()
    mask = segment_volume(zero_model, gray, torch.device("cpu"), image_size=32,
                          threshold=0.5, batch_size=2)
    assert mask.shape == gray.shape and not mask.any()


def test_classifier_forward_and_classification_threshold():
    config = load_config(ROOT / "configs" / "T_staging.yaml")
    model = build_classifier(config, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 2)
    probability = positive_probability(logits)
    assert 0.0 <= probability <= 1.0
    assert binary_prediction(0.5, 0.5) == 1
    assert binary_prediction(0.499999, 0.5) == 0


def test_window_postprocess_and_runs():
    volume = np.asarray([-1000.0, -125.0, 50.0, 225.0, 1000.0])
    gray = abdominal_window(volume)
    assert gray.tolist() == [0, 0, 127, 255, 255]
    area = np.asarray([0, 20, 0, 0, 30, 0, 0, 0, 40])
    assert z_runs(area, gap=2, min_area=20) == [(1, 4), (8, 8)]
    mask = np.zeros((12, 12, 10), dtype=bool)
    mask[2:6, 2:6, 2:5] = True
    mask[8:10, 8:10, 9] = True
    result, info = postprocess_mask(mask, gap=2, margin=2, min_area=2)
    assert result.sum() == 48
    assert info["main_run"] == [2, 4]


def test_key_slice_roi_and_boundary_clamp():
    gray = np.random.default_rng(42).integers(0, 256, size=(24, 32, 9), dtype=np.uint8)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[5:10, 7:12, 3] = True
    mask[4:13, 5:15, 5] = True
    key_z, fallback = select_key_slice(mask)
    assert key_z == 5 and not fallback
    roi, info = build_25d_roi(gray, mask, image_size=32)
    assert roi.shape == (3, 32, 32) and roi.dtype == np.uint8
    assert info["key_z"] == 5 and not info["empty_mask_fallback"]
    normalized = normalize_roi(roi)
    assert normalized.shape == (3, 32, 32) and np.isfinite(normalized).all()
    empty = np.zeros_like(mask)
    fallback_key, fallback = select_key_slice(empty)
    assert fallback_key == gray.shape[2] // 2 and fallback
    fallback_roi, fallback_info = build_25d_roi(gray, empty, image_size=32)
    assert fallback_roi.shape == (3, 32, 32) and fallback_info["empty_mask_fallback"]
    training_roi, training_info = build_25d_roi(gray, empty, image_size=32,
                                               key_z_gt=0, training=True)
    assert training_info["key_z"] == 0 and training_roi.shape == (3, 32, 32)
    boundary_mask = np.zeros_like(mask)
    boundary_mask[3:8, 4:9, 0] = True
    boundary_roi, boundary_info = build_25d_roi(gray, boundary_mask, image_size=32)
    assert boundary_info["key_z"] == 0
    assert np.array_equal(boundary_roi[0], boundary_roi[1])


def test_missing_weights_fail_explicitly():
    with tempfile.TemporaryDirectory() as directory:
        missing = Path(directory) / "absent.pt"
        try:
            load_state_dict(missing, torch.device("cpu"))
        except FileNotFoundError as error:
            assert "公开包不含训练权重" in str(error)
        else:
            raise AssertionError("缺失权重必须报错")


def main():
    tests = [
        test_segmentation_forward_and_threshold,
        test_classifier_forward_and_classification_threshold,
        test_window_postprocess_and_runs,
        test_key_slice_roi_and_boundary_clamp,
        test_missing_weights_fail_explicitly,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"SMOKE TEST PASSED: {len(tests)} tests")


if __name__ == "__main__":
    main()
