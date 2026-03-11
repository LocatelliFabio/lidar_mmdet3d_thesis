# cyclist_speed.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


CYCLIST_LABEL = 1


def points_in_oriented_box(
    points_xyz: np.ndarray,
    box7: np.ndarray,
    pad=(0.25, 0.25, 0.50),
) -> np.ndarray:
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(np.float64)

    center = np.array([cx, cy, cz], dtype=np.float64)
    half = np.array([dx, dy, dz], dtype=np.float64) / 2.0
    half += np.array(pad, dtype=np.float64)

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    local_pts = (points_xyz - center) @ rot

    mask = (
        (np.abs(local_pts[:, 0]) <= half[0]) &
        (np.abs(local_pts[:, 1]) <= half[1]) &
        (np.abs(local_pts[:, 2]) <= half[2])
    )
    return mask


def normalize_yaw_with_previous(box7: np.ndarray, prev_yaw_rad: float | None) -> np.ndarray:
    if box7 is None:
        return None

    out = box7.copy()
    yaw_deg = np.degrees(out[6])

    if prev_yaw_rad is not None:
        prev_deg = np.degrees(prev_yaw_rad)
        diff = prev_deg - yaw_deg

        if diff < -90.0:
            yaw_deg -= 180.0
        elif diff > 90.0:
            yaw_deg += 180.0

    out[6] = np.radians(yaw_deg)
    return out


def fix_second_cyclist_box(
    box7: np.ndarray,
    sensor_height_fix: float = 1.7 / 2.0,
) -> np.ndarray:
    b = box7.copy()

    b[2] = b[2] + sensor_height_fix

    dx, dy = b[3], b[4]
    b[3], b[4] = dy, dx

    b[6] = -b[6]

    return b


@dataclass
class CyclistSpeedState:
    detected: bool
    instant_kmh: float
    smooth_kmh: float
    mean_kmh: float
    max_kmh: float
    total_distance_m: float
    center: np.ndarray | None
    num_points: int
    score: float | None
    box7: np.ndarray | None


class RealTimeCyclistSpeedEstimator:
    def __init__(
        self,
        score_thr: float = 0.30,
        pad=(0.25, 0.25, 0.50),
        smoothing_window: int = 5,
        use_xy_only: bool = True,
        min_points_in_box: int = 5,
        max_reasonable_speed_kmh: float = 80.0,
    ):
        self.score_thr = score_thr
        self.pad = pad
        self.use_xy_only = use_xy_only
        self.min_points_in_box = min_points_in_box
        self.max_reasonable_speed_kmh = max_reasonable_speed_kmh

        self.prev_center = None
        self.prev_time = None
        self.prev_yaw = None

        self.total_distance_m = 0.0
        self.max_kmh = 0.0
        self.sum_kmh = 0.0
        self.num_speed_samples = 0

        self.speed_hist = deque(maxlen=smoothing_window)

    def _select_best_cyclist(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ):
        if boxes is None or len(boxes) == 0:
            return None, None

        mask = (labels == CYCLIST_LABEL) & (scores >= self.score_thr)
        if not np.any(mask):
            return None, None

        idxs = np.where(mask)[0]
        best_idx = idxs[np.argmax(scores[idxs])]

        return boxes[best_idx].copy(), float(scores[best_idx])

    def _compute_center_from_box_points(
        self,
        points_xyzi: np.ndarray,
        box7: np.ndarray,
    ):
        if points_xyzi is None or len(points_xyzi) == 0:
            return None, 0

        mask = points_in_oriented_box(points_xyzi[:, :3], box7, pad=self.pad)
        inside = points_xyzi[mask]

        if inside.shape[0] < self.min_points_in_box:
            return None, inside.shape[0]

        center = inside[:, :3].mean(axis=0).astype(np.float64)
        return center, inside.shape[0]

    def update(
        self,
        points_xyzi: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        timestamp: float | None = None,
    ) -> CyclistSpeedState:
        if timestamp is None:
            raise ValueError("RealTimeCyclistSpeedEstimator.update richiede il timestamp reale del frame")

        now = float(timestamp)

        best_box, best_score = self._select_best_cyclist(boxes, scores, labels)
        if best_box is None:
            return CyclistSpeedState(
                detected=False,
                instant_kmh=0.0,
                smooth_kmh=(float(np.mean(self.speed_hist)) if len(self.speed_hist) > 0 else 0.0),
                mean_kmh=(self.sum_kmh / self.num_speed_samples if self.num_speed_samples > 0 else 0.0),
                max_kmh=self.max_kmh,
                total_distance_m=self.total_distance_m,
                center=None,
                num_points=0,
                score=None,
                box7=None,
            )

        box7 = fix_second_cyclist_box(best_box)
        box7 = normalize_yaw_with_previous(box7, self.prev_yaw)

        center, npts = self._compute_center_from_box_points(points_xyzi, box7)

        if center is None:
            center = box7[:3].astype(np.float64)

        instant_kmh = 0.0

        if self.prev_center is not None and self.prev_time is not None:
            dt = now - self.prev_time

            if dt > 1e-3:
                if self.use_xy_only:
                    dist = np.linalg.norm(center[:2] - self.prev_center[:2])
                else:
                    dist = np.linalg.norm(center - self.prev_center)

                speed_kmh = (dist / dt) * 3.6

                if speed_kmh <= self.max_reasonable_speed_kmh:
                    instant_kmh = float(speed_kmh)
                    self.total_distance_m += float(dist)
                    self.max_kmh = max(self.max_kmh, instant_kmh)
                    self.sum_kmh += instant_kmh
                    self.num_speed_samples += 1
                    self.speed_hist.append(instant_kmh)

        smooth_kmh = float(np.mean(self.speed_hist)) if len(self.speed_hist) > 0 else 0.0
        mean_kmh = self.sum_kmh / self.num_speed_samples if self.num_speed_samples > 0 else 0.0

        self.prev_center = center
        self.prev_time = now
        self.prev_yaw = box7[6]

        return CyclistSpeedState(
            detected=True,
            instant_kmh=instant_kmh,
            smooth_kmh=smooth_kmh,
            mean_kmh=mean_kmh,
            max_kmh=self.max_kmh,
            total_distance_m=self.total_distance_m,
            center=center,
            num_points=npts,
            score=best_score,
            box7=box7,
        )