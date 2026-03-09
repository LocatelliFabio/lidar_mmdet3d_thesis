# rs_to_model_coords.py

import numpy as np


def rs_to_model_coords(points: np.ndarray) -> np.ndarray:
    out = points.copy()

    out[:, 0] = points[:, 2]
    out[:, 1] = -points[:, 0]
    out[:, 2] = -points[:, 1]

    if points.shape[1] > 3:
        out[:, 3] = points[:, 3]

    return out