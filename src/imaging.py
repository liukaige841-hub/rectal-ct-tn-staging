"""DICOM/NIfTI 输入读取；不在文件名或元数据中导出患者标识。"""
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    volume = np.asanyarray(image.dataobj)
    while volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"NIfTI 必须可还原为三维体数据：{path}")
    return np.ascontiguousarray(volume), image.affine


def _dicom_files(directory: Path) -> list[Path]:
    ignored = {"dicomdir", "lockfile", "version", "readme"}
    return [path for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name.lower() not in ignored and path.stat().st_size >= 128]


def load_dicom_series(directory: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """选择实例数最多的 CT 序列，应用 rescale，并构造 LPS→RAS affine。"""
    import pydicom

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"DICOM 输入目录不存在：{root}")
    metadata = []
    series_counts = {}
    for path in _dicom_files(root):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:
            continue
        if (dataset.get("Modality", "") != "CT" or "Rows" not in dataset
                or "Columns" not in dataset or "PixelSpacing" not in dataset):
            continue
        uid = str(dataset.get("SeriesInstanceUID", ""))
        series_counts[uid] = series_counts.get(uid, 0) + 1
        metadata.append((path, dataset))
    if not metadata:
        raise ValueError(f"目录中未找到含像素的 CT DICOM 序列：{root}")
    selected_uid = max(series_counts, key=series_counts.get)
    metadata = [(path, ds) for path, ds in metadata
                if str(ds.get("SeriesInstanceUID", "")) == selected_uid]
    acquisition_counts = {}
    for _, dataset in metadata:
        acquisition = str(dataset.get("AcquisitionNumber", ""))
        acquisition_counts[acquisition] = acquisition_counts.get(acquisition, 0) + 1
    if len(acquisition_counts) > 1:
        selected_acquisition = ("2" if "2" in acquisition_counts
                                else max(acquisition_counts, key=acquisition_counts.get))
        metadata = [(path, ds) for path, ds in metadata
                    if str(ds.get("AcquisitionNumber", "")) == selected_acquisition]

    def geometry(ds):
        orientation = np.asarray(ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
        row, column = orientation[:3], orientation[3:]
        normal = np.cross(row, column)
        position = np.asarray(ds.get("ImagePositionPatient", [0, 0, 0]), dtype=float)
        return row, column, normal, position

    metadata.sort(key=lambda item: float(np.dot(geometry(item[1])[3], geometry(item[1])[2]))
                  if item[1].get("ImagePositionPatient") is not None
                  else float(item[1].get("InstanceNumber", 0)))
    slices = []
    for path, _ in metadata:
        dataset = pydicom.dcmread(str(path))
        pixel = dataset.pixel_array.astype(np.float32)
        slices.append(pixel * float(dataset.get("RescaleSlope", 1.0))
                      + float(dataset.get("RescaleIntercept", 0.0)))
    shapes = {slice_.shape for slice_ in slices}
    if len(shapes) != 1:
        raise ValueError(f"同一 DICOM 序列的层尺寸不一致：{sorted(shapes)}")
    volume = np.stack(slices, axis=0).transpose(2, 1, 0).astype(np.float32)
    first = metadata[0][1]
    row, column, normal, position = geometry(first)
    spacing = [float(value) for value in first.PixelSpacing]
    thickness = float(first.get("SliceThickness", 1.0))
    positions = [float(np.dot(geometry(ds)[3], geometry(ds)[2])) for _, ds in metadata]
    differences = np.abs(np.diff(positions))
    valid = differences[differences > 1e-3]
    spacing_z = float(np.median(valid)) if len(valid) else thickness
    if spacing_z <= 0:
        raise ValueError("无法确定有效的 DICOM 层间距")
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = row * spacing[1]
    affine_lps[:3, 1] = column * spacing[0]
    affine_lps[:3, 2] = normal * spacing_z
    affine_lps[:3, 3] = position
    affine_ras = affine_lps.copy()
    affine_ras[0, :] *= -1
    affine_ras[1, :] *= -1
    canonical = nib.as_closest_canonical(nib.Nifti1Image(volume, affine_ras))
    return np.ascontiguousarray(np.asanyarray(canonical.dataobj)), canonical.affine


def load_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    input_path = Path(path).expanduser()
    if input_path.is_dir():
        return load_dicom_series(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入影像不存在：{input_path}")
    name = input_path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ValueError("文件输入仅支持 .nii 或 .nii.gz；DICOM 请传入序列目录")
    return load_nifti(input_path)
