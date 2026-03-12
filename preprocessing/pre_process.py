# preprocessing/pre_process.py

import numpy as np
from typing import Optional

try:
    import linefit  # pip install linefit
    _LINEFIT_AVAILABLE = True
except ImportError:
    linefit = None
    _LINEFIT_AVAILABLE = False


def remove_ground_grid(
    points: np.ndarray,
    cell: float = 0.5,
    thresh: float = 0.05,
    pc_range=(0, -40, -3, 70.4, 40, 1),
) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    xmin, ymin, *_ = pc_range
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    gx = np.floor((x - xmin) / cell).astype(np.int32)
    gy = np.floor((y - ymin) / cell).astype(np.int32)
    keys = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xFFFFFFFF)

    order = np.argsort(keys)
    keys_s = keys[order]
    z_s = z[order]

    change = np.ones_like(keys_s, dtype=bool)
    change[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.nonzero(change)[0]

    zmin = np.minimum.reduceat(z_s, starts)
    counts = np.diff(np.append(starts, len(keys_s)))
    zmin_per_point_sorted = np.repeat(zmin, counts)

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    zmin_per_point = zmin_per_point_sorted[inv]

    keep = (z - zmin_per_point) > thresh
    return points[keep].astype(np.float32, copy=False)


def grid_average_downsample_xyzi(
    points: np.ndarray,
    voxel: float = 0.08,
    pc_range=(0, -40, -3, 70.4, 40, 1),
) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    xmin, ymin, zmin, *_ = pc_range

    q = np.floor(
        (points[:, :3] - np.array([xmin, ymin, zmin], dtype=np.float32)) / voxel
    ).astype(np.int32)

    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
    key = (
        (qx.astype(np.int64) << 42)
        | (qy.astype(np.int64) << 21)
        | qz.astype(np.int64)
    )

    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    n = counts.size

    out = np.empty((n, 4), dtype=np.float32)
    out[:, 0] = np.bincount(inv, weights=points[:, 0], minlength=n) / counts
    out[:, 1] = np.bincount(inv, weights=points[:, 1], minlength=n) / counts
    out[:, 2] = np.bincount(inv, weights=points[:, 2], minlength=n) / counts
    out[:, 3] = np.bincount(inv, weights=points[:, 3], minlength=n) / counts
    return out.astype(np.float32, copy=False)


def voxel_occupancy_denoise(
    points: np.ndarray,
    voxel: float = 0.25,
    min_pts: int = 2,
    pc_range=(0, -40, -3, 70.4, 40, 1),
) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    xmin, ymin, zmin, *_ = pc_range

    q = np.floor(
        (points[:, :3] - np.array([xmin, ymin, zmin], dtype=np.float32)) / voxel
    ).astype(np.int32)

    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
    key = (
        (qx.astype(np.int64) << 42)
        | (qy.astype(np.int64) << 21)
        | qz.astype(np.int64)
    )

    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep = counts[inv] >= min_pts
    return points[keep].astype(np.float32, copy=False)


def _call_linefit_xyz(
    xyz: np.ndarray,
    config_path: Optional[str] = None,
) -> np.ndarray:
    if not _LINEFIT_AVAILABLE:
        raise ImportError(
            "Il pacchetto 'linefit' non è installato. "
            "Installa con: pip install linefit"
        )

    if config_path is None:
        seg = linefit.ground_seg()
    else:
        seg = linefit.ground_seg(config_path)

    mask = seg.run(xyz)
    return np.asarray(mask).reshape(-1).astype(bool)


def remove_ground_linefit(
    points: np.ndarray,
    config_path: Optional[str] = None,
    return_ground_mask: bool = False,
):
    if points is None or points.shape[0] == 0:
        empty = np.empty((0, 4), dtype=np.float32)
        if return_ground_mask:
            return empty, np.zeros((0,), dtype=bool)
        return empty

    pts = np.asarray(points, dtype=np.float32)
    xyz = pts[:, :3]

    ground_mask = _call_linefit_xyz(
        xyz,
        config_path=config_path,
    )

    if ground_mask.shape[0] != pts.shape[0]:
        raise RuntimeError(
            f"LineFit ha restituito una mask di lunghezza {ground_mask.shape[0]}, "
            f"ma i punti sono {pts.shape[0]}."
        )

    non_ground = pts[~ground_mask].astype(np.float32, copy=False)

    if return_ground_mask:
        return non_ground, ground_mask
    return non_ground


def preprocess_raw_for_second(
    points: np.ndarray,
    pc_range=(0, -40, -3, 70.4, 40, 1),
    ds_voxel: float = 0.08,
    denoise: bool = False,
    den_voxel: float = 0.35,
    den_min_pts: int = 2,
    ground_method: str = "grid",   # "grid", "linefit", "none"
    ground_cell: float = 0.5,
    ground_thresh: float = 0.07,
    linefit_cfg: Optional[dict] = None,
) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Atteso array Nx3 o Nx4+, ricevuto shape={points.shape}")

    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    if points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    points = points.astype(np.float32, copy=False)

    if points.shape[1] < 4:
        intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack((points[:, :3], intensity))
    elif points.shape[1] > 4:
        points = points[:, :4]

    points[:, 3] = np.clip(points[:, 3], 0, 255) / 255.0

    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    keep = (
        (x >= xmin) & (x <= xmax) &
        (y >= ymin) & (y <= ymax) &
        (z >= zmin) & (z <= zmax)
    )
    points = points[keep]

    if points.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    points = grid_average_downsample_xyzi(
        points,
        voxel=ds_voxel,
        pc_range=pc_range,
    )

    if denoise and points.shape[0] > 0:
        points = voxel_occupancy_denoise(
            points,
            voxel=den_voxel,
            min_pts=den_min_pts,
            pc_range=pc_range,
        )

    if points.shape[0] > 0:
        method = str(ground_method).strip().lower()

        if method == "grid":
            points = remove_ground_grid(
                points,
                cell=ground_cell,
                thresh=ground_thresh,
                pc_range=pc_range,
            )
        elif method == "linefit":
            cfg = {} if linefit_cfg is None else dict(linefit_cfg)
            points = remove_ground_linefit(points, **cfg)
        elif method in ("none", "off", "disabled"):
            pass
        else:
            raise ValueError(
                f"ground_method='{ground_method}' non supportato. "
                "Usa: 'grid', 'linefit' oppure 'none'."
            )

    return points.astype(np.float32, copy=False)