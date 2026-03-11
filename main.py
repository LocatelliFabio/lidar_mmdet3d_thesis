# main.py

from time import sleep, monotonic
import numpy as np

from config import (
    MODEL_CONFIG,
    MODEL_CHECKPOINT,
    MODEL_DEVICE,
    PREPROCESS_CFG,
    SCORE_THR,
    LIDAR_CFG,
    VIEWER_CFG,
    LOOP_SLEEP_SEC,
    ENABLE_DEBUG_LOGS,
    TRACKER_CFG,
)

from thread_safe_buffer import LatestValueBuffer
from rs_lidar_stream import RSLidarStream
from preprocessing.pre_process import preprocess_raw_for_second
from detector import SecondDetector
from live_viewer import LiveViewer3D
from rs_to_model_coords import rs_to_model_coords
from cyclist_tracker import RealTimeCyclistTracker


def log_cloud_stats(name: str, points: np.ndarray) -> None:
    if not ENABLE_DEBUG_LOGS:
        return
    if points is None or len(points) == 0:
        print(f"{name}: empty")
        return

    xyz = points[:, :3]
    print(f"{name} min xyz: {xyz.min(axis=0)}")
    print(f"{name} max xyz: {xyz.max(axis=0)}")
    print(f"{name} shape: {points.shape}")


def main():
    frame_buffer = LatestValueBuffer()

    lidar = RSLidarStream(frame_buffer=frame_buffer, lidar_cfg=LIDAR_CFG)
    if not lidar.init():
        print("Driver init failed")
        return

    detector = SecondDetector(
        config_path=MODEL_CONFIG,
        checkpoint_path=MODEL_CHECKPOINT,
        device=MODEL_DEVICE,
    )

    viewer = LiveViewer3D(
        window_name=VIEWER_CFG["window_name"],
        width=VIEWER_CFG["width"],
        height=VIEWER_CFG["height"],
        point_size=VIEWER_CFG["point_size"],
    )

    tracker = RealTimeCyclistTracker(**TRACKER_CFG)

    last_processed_seq = -1
    last_status_print = 0.0
    status_print_period = 0.2

    lidar.start()

    try:
        while True:
            raw_points, seq, frame_ts = frame_buffer.get_copy()

            if raw_points is None or raw_points.shape[0] == 0:
                viewer.spin_once()
                sleep(LOOP_SLEEP_SEC)
                continue

            # evita di rielaborare lo stesso frame più volte
            if seq == last_processed_seq:
                viewer.spin_once()
                sleep(LOOP_SLEEP_SEC)
                continue

            last_processed_seq = seq

            model_points = rs_to_model_coords(raw_points)
            processed_points = preprocess_raw_for_second(model_points, **PREPROCESS_CFG)

            log_cloud_stats("RAW", raw_points)
            log_cloud_stats("MODEL", model_points)
            log_cloud_stats("PROCESSED", processed_points)

            if processed_points.shape[0] == 0:
                viewer.update(
                    model_points,
                    np.empty((0, 7), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )
                sleep(LOOP_SLEEP_SEC)
                continue

            result = detector.infer(processed_points)
            boxes, scores, labels = detector.extract_predictions(
                result,
                score_thr=SCORE_THR,
            )

            track_state = tracker.update(
                boxes=boxes,
                scores=scores,
                labels=labels,
                timestamp=frame_ts,
            )

            viewer.update(model_points, boxes, labels)

            now = monotonic()
            if now - last_status_print >= status_print_period:
                if track_state.detected:
                    print(
                        f"Cyclist "
                        f"| score={track_state.score:.2f} "
                        f"| v={track_state.instant_kmh:6.2f} km/h "
                        f"| v_smooth={track_state.smooth_kmh:6.2f} km/h "
                        f"| v_mean={track_state.mean_kmh:6.2f} km/h "
                        f"| v_max={track_state.max_kmh:6.2f} km/h "
                        f"| dist={track_state.total_distance_m:7.2f} m "
                        f"| match_max={track_state.allowed_match_distance_m:5.2f} m",
                        end="\r",
                    )
                else:
                    print(
                        f"Cyclist | no detection "
                        f"| v_smooth={track_state.smooth_kmh:6.2f} km/h "
                        f"| v_max={track_state.max_kmh:6.2f} km/h "
                        f"| dist={track_state.total_distance_m:7.2f} m "
                        f"| match_max={track_state.allowed_match_distance_m:5.2f} m",
                        end="\r",
                    )

                last_status_print = now

            sleep(LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        print("\nChiusura...")

    finally:
        lidar.stop()
        viewer.close()


if __name__ == "__main__":
    main()