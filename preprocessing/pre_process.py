import numpy as np

def remove_ground_grid(points: np.ndarray, cell=0.5, thresh=0.05, pc_range=(0,-40,-3,70.4,40,1)) -> np.ndarray:
    xmin, ymin, *_ = pc_range
    x, y, z = points[:,0], points[:,1], points[:,2]

    gx = np.floor((x - xmin) / cell).astype(np.int32)
    gy = np.floor((y - ymin) / cell).astype(np.int32)
    keys = (gx.astype(np.int64) << 32) | (gy.astype(np.int64) & 0xffffffff)

    # ordina per key e poi prendi min z per key
    order = np.argsort(keys)
    keys_s = keys[order]
    z_s = z[order]

    # trova inizio gruppi
    change = np.ones_like(keys_s, dtype=bool)
    change[1:] = keys_s[1:] != keys_s[:-1]
    starts = np.nonzero(change)[0]

    # min z per gruppo (cella)
    zmin = np.minimum.reduceat(z_s, starts)

    # mappa zmin a ogni punto (nel sorted order)
    # ripeti ogni zmin per la lunghezza del gruppo
    counts = np.diff(np.append(starts, len(keys_s)))
    zmin_per_point_sorted = np.repeat(zmin, counts)

    # torna all’ordine originale
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    zmin_per_point = zmin_per_point_sorted[inv]

    keep = (z - zmin_per_point) > thresh
    return points[keep]


def grid_average_downsample_xyzi(points: np.ndarray,
                                      voxel=0.08,
                                      pc_range=(0, -40, -3, 70.4, 40, 1)) -> np.ndarray:
    xmin, ymin, zmin, *_ = pc_range
    q = np.floor((points[:, :3] - np.array([xmin, ymin, zmin], dtype=np.float32)) / voxel).astype(np.int32)
    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
    key = (qx.astype(np.int64) << 42) | (qy.astype(np.int64) << 21) | qz.astype(np.int64)

    # gruppi senza sort
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    n = counts.size

    out = np.empty((n, 4), dtype=np.float32)
    out[:,0] = np.bincount(inv, weights=points[:,0], minlength=n) / counts
    out[:,1] = np.bincount(inv, weights=points[:,1], minlength=n) / counts
    out[:,2] = np.bincount(inv, weights=points[:,2], minlength=n) / counts
    out[:,3] = np.bincount(inv, weights=points[:,3], minlength=n) / counts
    return out


def voxel_occupancy_denoise(points: np.ndarray,
                            voxel=0.25,
                            min_pts=2,
                            pc_range=(0, -40, -3, 70.4, 40, 1)) -> np.ndarray:
    
    xmin, ymin, zmin, *_ = pc_range
    q = np.floor(
        (points[:, :3] - np.array([xmin, ymin, zmin], dtype=np.float32)) / voxel
    ).astype(np.int32)

    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]

    key = (qx.astype(np.int64) << 42) | (qy.astype(np.int64) << 21) | qz.astype(np.int64)

    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)

    keep = counts[inv] >= min_pts
    return points[keep]


def preprocess_raw_for_second(points: np.ndarray,
                              pc_range=(0, -40, -3, 70.4, 40, 1),
                              ds_voxel=0.08,
                              denoise=False,
                              den_voxel=0.35, den_min_pts=2,
                              ground_cell=0.5,
                              ground_thresh=0.07):
    # 0) valid
    points = points[np.isfinite(points[:, :3]).all(axis=1)]

    # 1) intensity -> [0,1]
    points = points.astype(np.float32, copy=False)
    points[:, 3] = np.clip(points[:, 3], 0, 255) / 255.0

    # 2) crop to model range
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
    points = points[m]

    # 3) downsample gridAverage (MATLAB-like)
    points = grid_average_downsample_xyzi(points, voxel=ds_voxel, pc_range=pc_range)

    # 4) denoise (cheap, optional)
    if denoise:
        points = voxel_occupancy_denoise(points, voxel=den_voxel, min_pts=den_min_pts, pc_range=pc_range)

    # 5) ground removal (MATLAB-like threshold)
    points = remove_ground_grid(points, cell=ground_cell, thresh=ground_thresh, pc_range=pc_range)

    return points
