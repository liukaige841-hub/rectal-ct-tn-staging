"""从后处理掩膜构建训练侧三通道 2.5D ROI 与分类清单。"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.config import load_config
from src.imaging import load_nifti
from src.roi import build_25d_roi, save_roi_pngs
from src.segmentation import abdominal_window


def build(manifest_path: Path, output_dir: Path, config: dict):
    frame = pd.read_csv(manifest_path, dtype={"sample_id": str})
    required = {"sample_id", "image_path", "mask_path", "label", "split", "key_z"}
    if required - set(frame.columns):
        raise ValueError(f"输入清单缺少列：{sorted(required - set(frame.columns))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = config["data"]
    rows = []
    for _, row in frame.iterrows():
        image_path = Path(row["image_path"]).expanduser()
        mask_path = Path(row["mask_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not mask_path.is_absolute():
            mask_path = manifest_path.parent / mask_path
        volume, _ = load_nifti(image_path)
        mask_image = nib.as_closest_canonical(nib.load(str(mask_path)))
        mask = np.asanyarray(mask_image.dataobj)
        while mask.ndim > 3:
            mask = mask[..., 0]
        if volume.shape != mask.shape:
            raise ValueError(f"样本 {row['sample_id']} 的影像与掩膜形状不一致")
        gray = abdominal_window(volume, 50.0, 350.0)
        roi, info = build_25d_roi(
            gray, mask > 0, image_size=int(data["image_size"]),
            neighbor_slices=int(data["neighbor_slices"]), bbox_window=int(data["bbox_window"]),
            margin_ratio=float(data["margin_ratio"]),
            key_z_gt=int(row["key_z"]), training=True,
        )
        stem = output_dir / str(row["sample_id"]) / "roi"
        paths = save_roi_pngs(roi, stem)
        rows.append({"sample_id": str(row["sample_id"]),
                     "image_path": paths[1].relative_to(output_dir).as_posix(),
                     "label": int(row["label"]), "split": row["split"],
                     "key_z": info["key_z"], "empty_mask_fallback": info["empty_mask_fallback"]})
        print(f"已处理样本 {row['sample_id']}", flush=True)
    labels_path = output_dir / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False, encoding="utf-8")
    return labels_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="CSV：sample_id,image_path,mask_path,label,split,key_z")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True,
                        help="configs/T_staging.yaml 或 configs/N_staging.yaml")
    args = parser.parse_args()
    result = build(Path(args.manifest).expanduser().resolve(),
                   Path(args.output_dir).expanduser().resolve(), load_config(args.config))
    print(f"分类清单：{result}")


if __name__ == "__main__":
    main()
