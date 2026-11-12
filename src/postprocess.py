"""分割掩膜的固定三维后处理。"""
import numpy as np
import scipy.ndimage as ndi


def z_runs(area: np.ndarray, gap: int = 2, min_area: int = 20) -> list[tuple[int, int]]:
    positive = np.where(np.asarray(area) >= min_area)[0]
    if len(positive) == 0:
        return []
    runs = []
    start = previous = int(positive[0])
    for z_value in positive[1:]:
        z = int(z_value)
        if z - previous <= gap + 1:
            previous = z
        else:
            runs.append((start, previous))
            start = previous = z
    runs.append((start, previous))
    return runs


def largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    coordinates = np.where(mask)
    if len(coordinates[0]) == 0:
        return np.zeros_like(mask, dtype=bool), 0
    lower = [int(axis.min()) for axis in coordinates]
    upper = [int(axis.max()) + 1 for axis in coordinates]
    crop = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    labels, count = ndi.label(crop, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= 1:
        return mask.astype(bool, copy=True), int(count)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    output = np.zeros_like(mask, dtype=bool)
    output[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]] = labels == sizes.argmax()
    return output, int(count)


def postprocess_mask(
    mask: np.ndarray, gap: int = 2, margin: int = 2, min_area: int = 20
) -> tuple[np.ndarray, dict]:
    """最大 z 连续段外延后，保留段内 26 邻接最大三维连通域。"""
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 3:
        raise ValueError(f"掩膜必须为三维，实际为 {binary.shape}")
    area = binary.sum(axis=(0, 1))
    runs = z_runs(area, gap=gap, min_area=min_area)
    if not runs:
        return np.zeros_like(binary), {"n_runs": 0, "n_components": 0, "main_run": None}
    main = max(runs, key=lambda run: int(binary[:, :, run[0]:run[1] + 1].sum()))
    z0 = max(0, main[0] - margin)
    z1 = min(binary.shape[2] - 1, main[1] + margin)
    kept = np.zeros_like(binary)
    kept[:, :, z0:z1 + 1] = binary[:, :, z0:z1 + 1]
    result, count = largest_connected_component(kept)
    return result, {"n_runs": len(runs), "n_components": count, "main_run": list(main)}
