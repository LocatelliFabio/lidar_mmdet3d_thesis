# rs_to_model_coords.py

import numpy as np

def rs_to_model_coords(points: np.ndarray) -> np.ndarray:

    '''
    Nel driver rs_driver_python:
    unchecked_ref(row, 0) = -point_cloud.points[row].y;
    unchecked_ref(row, 1) = -point_cloud.points[row].z;
    unchecked_ref(row, 2) = point_cloud.points[row].x;
    unchecked_ref(row, 3) = (float)point_cloud.points[row].intensity;
    '''


    out = points.copy()

    out[:, 0] = points[:, 2]
    out[:, 1] = -points[:, 0]
    out[:, 2] = -points[:, 1]

    if points.shape[1] > 3:
        out[:, 3] = points[:, 3]

    return out