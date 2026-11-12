"""从通用 NIfTI 清单构建最终 2D 分割切片索引。"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from src.config import load_config
from src.imaging import load_nifti
from src.segmentation import abdominal_window


def build(manifest_path: Path, output_dir: Path, config: dict):
    frame = pd.read_csv(manifest_path, dtype={"sample_id": str})
    required = {"sample_id", "image_path", "mask_path", "split"}
    if required - set(frame.columns):
        raise ValueError(f"输入清单缺少列：{sorted(required - set(frame.columns))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    size = int(config["data"]["image_size"])
    margin = int(config["data"].get("negative_margin_slices", 60))
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
        mask = mask > 0
        if not mask.any():
            raise ValueError(f"样本 {row['sample_id']} 的标注掩膜为空")
        gray = abdominal_window(volume, float(config["data"]["window_center"]),
                                float(config["data"]["window_width"]))
        area = mask.sum(axis=(0, 1))
        positive = np.nonzero(area)[0]
        lower = max(0, int(positive.min()) - margin)
        upper = min(mask.shape[2] - 1, int(positive.max()) + margin)
        positive_set = set(int(z) for z in positive)
        negative_pool = [z for z in range(lower, upper + 1) if z not in positive_set]
        if len(negative_pool) > len(positive_set):
            indices = np.linspace(0, len(negative_pool) - 1, len(positive_set)).round().astype(int)
            negative = sorted({negative_pool[index] for index in indices})
        else:
            negative = negative_pool
        sample_dir = output_dir / str(row["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        for z in sorted(positive_set | set(negative)):
            name = f"z{z:04d}.png"
            image = Image.fromarray(np.rot90(gray[:, :, z]), mode="L").resize(
                (size, size), Image.Resampling.BILINEAR)
            mask_png = Image.fromarray(np.rot90(mask[:, :, z]).astype(np.uint8) * 255,
                                       mode="L").resize((size, size), Image.Resampling.NEAREST)
            image.save(sample_dir / name)
            mask_png.save(sample_dir / f"z{z:04d}_mask.png")
            rows.append({"sample_id": str(row["sample_id"]), "z": z,
                         "image_path": f"{row['sample_id']}/{name}",
                         "mask_path": f"{row['sample_id']}/z{z:04d}_mask.png",
                         "has_mask": int(z in positive_set), "split": row["split"]})
        print(f"已处理样本 {row['sample_id']}", flush=True)
    index_path = output_dir / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False, encoding="utf-8")
    return index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="CSV：sample_id,image_path,mask_path,split")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/segmentation.yaml")
    args = parser.parse_args()
    result = build(Path(args.manifest).expanduser().resolve(),
                   Path(args.output_dir).expanduser().resolve(), load_config(args.config))
    print(f"分割索引：{result}")


if __name__ == "__main__":
    main()
