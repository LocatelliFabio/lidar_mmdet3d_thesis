# main.py

from time import sleep, perf_counter
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
    SPEED_CFG,
)

from thread_safe_buffer import LatestValueBuffer
from rs_lidar_stream import RSLidarStream
from preprocessing.pre_process import preprocess_raw_for_second
from detector import SecondDetector
from live_viewer import LiveViewer3D
from rs_to_model_coords import rs_to_model_coords
from cyclist_speed import RealTimeCyclistSpeedEstimator


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
        camera_position=VIEWER_CFG["camera_position"],
        camera_lookat=VIEWER_CFG["camera_lookat"],
    )

    speed_estimator = RealTimeCyclistSpeedEstimator(**SPEED_CFG)

    last_seq = -1
    last_frame_ts = None

    lidar.start()

    i = 0

    try:
        while True:
            raw_points, frame_ts, seq = frame_buffer.get_latest_copy()

            if raw_points is None or frame_ts is None or seq == last_seq:
                viewer.spin_once()
                sleep(LOOP_SLEEP_SEC)
                continue
            
            last_seq = seq

            if i < 1:
                i = i + 1
                continue
            else:
                i = 0


            t0 = perf_counter()

            model_points = rs_to_model_coords(raw_points)
            t1 = perf_counter()

            processed_points = preprocess_raw_for_second(model_points, **PREPROCESS_CFG)
            t2 = perf_counter()

            log_cloud_stats("RAW", raw_points)
            log_cloud_stats("MODEL", model_points)
            log_cloud_stats("PROCESSED", processed_points)

            if processed_points.shape[0] == 0:
                empty_boxes = np.empty((0, 7), dtype=np.float32)
                empty_labels = np.empty((0,), dtype=np.int64)

                viewer.update(model_points, empty_boxes, empty_labels)
                t3 = perf_counter()

                frame_period_ms = 0.0 if last_frame_ts is None else (frame_ts - last_frame_ts) * 1000.0
                total_ms = (t3 - t0) * 1000.0

                print(
                    f"Frame seq={seq:06d} | pts_raw={raw_points.shape[0]:6d} "
                    f"| pts_proc=0 | dt_frame={frame_period_ms:7.2f} ms "
                    f"| coord={(t1 - t0) * 1000.0:7.2f} ms "
                    f"| prep={(t2 - t1) * 1000.0:7.2f} ms "
                    f"| infer={0.0:7.2f} ms "
                    f"| speed={0.0:7.2f} ms "
                    f"| view={(t3 - t2) * 1000.0:7.2f} ms "
                    f"| total={total_ms:7.2f} ms"
                )

                last_frame_ts = frame_ts
                continue

            result = detector.infer(processed_points)
            t3 = perf_counter()

            boxes, scores, labels = detector.extract_predictions(
                result,
                score_thr=SCORE_THR,
            )
            t4 = perf_counter()

            speed_state = speed_estimator.update(
                boxes=boxes,
                scores=scores,
                labels=labels,
                timestamp=frame_ts,   # timestamp reale del frame
            )
            t5 = perf_counter()

            viewer.update(model_points, boxes, labels)
            t6 = perf_counter()

            frame_period_ms = 0.0 if last_frame_ts is None else (frame_ts - last_frame_ts) * 1000.0
            total_ms = (t6 - t0) * 1000.0

            print(
                f"Frame seq={seq:06d} | pts_raw={raw_points.shape[0]:6d} "
                f"| pts_proc={processed_points.shape[0]:6d} "
                f"| det={len(boxes):2d} "
                f"| dt_frame={frame_period_ms:7.2f} ms "
                f"| coord={(t1 - t0) * 1000.0:7.2f} ms "
                f"| prep={(t2 - t1) * 1000.0:7.2f} ms "
                f"| infer={(t3 - t2) * 1000.0:7.2f} ms "
                f"| post={(t4 - t3) * 1000.0:7.2f} ms "
                f"| speed={(t5 - t4) * 1000.0:7.2f} ms "
                f"| view={(t6 - t5) * 1000.0:7.2f} ms "
                f"| total={total_ms:7.2f} ms"
            )

            if speed_state.detected:
                print(
                    f"Cyclist | score={speed_state.score:.2f} "
                    f"| v={speed_state.instant_kmh:6.2f} km/h "
                    f"| v_smooth={speed_state.smooth_kmh:6.2f} km/h "
                    f"| v_max={speed_state.max_kmh:6.2f} km/h "
                    f"| dist={speed_state.total_distance_m:7.2f} m"
                )
            else:
                print(
                    "Cyclist | no detection"
                    f" | v_smooth={speed_state.smooth_kmh:6.2f} km/h"
                    f" | v_max={speed_state.max_kmh:6.2f} km/h"
                    f" | dist={speed_state.total_distance_m:7.2f} m"
                )

            last_frame_ts = frame_ts

    except KeyboardInterrupt:
        print("\nChiusura...")

    finally:
        lidar.stop()
        viewer.close()


if __name__ == "__main__":
    main()