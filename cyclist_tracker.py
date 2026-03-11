# cyclist_tracker.py

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import monotonic

import numpy as np


CYCLIST_LABEL = 1


def fix_second_cyclist_box(
    box7: np.ndarray,
    sensor_height_fix: float = 1.7 / 2.0,
) -> np.ndarray:
    """
    Adatta la bbox Cyclist prodotta da SECOND alla convenzione usata
    nella tua pipeline.
    """
    b = box7.copy()

    b[2] = b[2] + sensor_height_fix

    dx, dy = b[3], b[4]
    b[3], b[4] = dy, dx

    b[6] = -b[6]

    return b


def normalize_yaw_with_previous(box7: np.ndarray, prev_yaw_rad: float | None) -> np.ndarray:
    """
    Evita flip di 180° tra frame consecutivi.
    """
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


def box_bottom_center_xy(box7: np.ndarray) -> np.ndarray:
    return box7[:2].astype(np.float64)


def box_geometric_center_xyz(box7: np.ndarray) -> np.ndarray:
    c = box7[:3].astype(np.float64).copy()
    c[2] += box7[5] / 2.0
    return c


@dataclass
class CyclistTrackState:
    detected: bool
    track_id: int | None
    score: float | None
    instant_kmh: float
    smooth_kmh: float
    mean_kmh: float
    max_kmh: float
    total_distance_m: float
    center_xyz: np.ndarray | None
    box7: np.ndarray | None


class RealTimeCyclistTracker:
    def __init__(
        self,
        score_thr: float = 0.40,
        smoothing_window: int = 5,
        max_reasonable_speed_kmh: float = 80.0,
        max_match_distance_m: float = 4.0,
        max_missed_frames: int = 5,
        prefer_high_score_when_unlocked: bool = True,
    ):
        self.score_thr = score_thr
        self.max_reasonable_speed_kmh = max_reasonable_speed_kmh
        self.max_match_distance_m = max_match_distance_m
        self.max_missed_frames = max_missed_frames
        self.prefer_high_score_when_unlocked = prefer_high_score_when_unlocked

        self.prev_center_xy = None
        self.prev_center_xyz = None
        self.prev_time = None
        self.prev_yaw = None

        self.total_distance_m = 0.0
        self.max_kmh = 0.0
        self.sum_kmh = 0.0
        self.num_speed_samples = 0
        self.speed_hist = deque(maxlen=smoothing_window)

        self.track_id = 1
        self.missed_frames = 0
        self.is_locked = False

    def _candidate_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ):
        if boxes is None or len(boxes) == 0:
            return []

        mask = (labels == CYCLIST_LABEL) & (scores >= self.score_thr)
        idxs = np.where(mask)[0]

        candidates = []
        for i in idxs:
            box = fix_second_cyclist_box(boxes[i])
            box = normalize_yaw_with_previous(box, self.prev_yaw)
            candidates.append((box, float(scores[i]), int(i)))

        return candidates

    def _pick_candidate(
        self,
        candidates,
    ):
        if len(candidates) == 0:
            return None, None

        if (not self.is_locked) or (self.prev_center_xy is None):
            if self.prefer_high_score_when_unlocked:
                best = max(candidates, key=lambda x: x[1])
            else:
                best = candidates[0]
            return best[0], best[1]

        # tracking per vicinanza alla posizione precedente
        prev_xy = self.prev_center_xy
        best_box = None
        best_score = None
        best_cost = None

        for box, score, _ in candidates:
            cxy = box_bottom_center_xy(box)
            dist = np.linalg.norm(cxy - prev_xy)

            if dist > self.max_match_distance_m:
                continue

            # costo: privilegia vicinanza, ma premia anche score alto
            cost = dist - 0.25 * score

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_box = box
                best_score = score

        if best_box is None:
            return None, None

        return best_box, best_score

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        timestamp: float | None = None,
    ) -> CyclistTrackState:
        now = monotonic() if timestamp is None else float(timestamp)

        candidates = self._candidate_boxes(boxes, scores, labels)
        box7, score = self._pick_candidate(candidates)

        if box7 is None:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.is_locked = False
                self.prev_center_xy = None
                self.prev_center_xyz = None
                self.prev_time = None
                self.prev_yaw = None

            return CyclistTrackState(
                detected=False,
                track_id=(self.track_id if self.is_locked else None),
                score=None,
                instant_kmh=0.0,
                smooth_kmh=(self.speed_hist[-1] if len(self.speed_hist) > 0 else 0.0),
                mean_kmh=(self.sum_kmh / self.num_speed_samples if self.num_speed_samples > 0 else 0.0),
                max_kmh=self.max_kmh,
                total_distance_m=self.total_distance_m,
                center_xyz=None,
                box7=None,
            )

        self.missed_frames = 0
        self.is_locked = True

        center_xy = box_bottom_center_xy(box7)
        center_xyz = box_geometric_center_xyz(box7)

        instant_kmh = 0.0

        if self.prev_center_xy is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 1e-3:
                dist = np.linalg.norm(center_xy - self.prev_center_xy)
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

        self.prev_center_xy = center_xy
        self.prev_center_xyz = center_xyz
        self.prev_time = now
        self.prev_yaw = box7[6]

        return CyclistTrackState(
            detected=True,
            track_id=self.track_id,
            score=score,
            instant_kmh=instant_kmh,
            smooth_kmh=smooth_kmh,
            mean_kmh=mean_kmh,
            max_kmh=self.max_kmh,
            total_distance_m=self.total_distance_m,
            center_xyz=center_xyz,
            box7=box7,
        )