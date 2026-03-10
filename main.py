# main.py

from time import sleep
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
)

from thread_safe_buffer import LatestValueBuffer
from rs_lidar_stream import RSLidarStream
from preprocessing.pre_process import preprocess_raw_for_second
from detector import SecondDetector
from live_viewer import LiveViewer3D
from rs_to_model_coords import rs_to_model_coords


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

    lidar.start()

    try:
        while True:
            raw_points = frame_buffer.get_copy()

            if raw_points is None or raw_points.shape[0] == 0:
                viewer.spin_once()
                sleep(LOOP_SLEEP_SEC)
                continue

            model_points = rs_to_model_coords(raw_points)

            # dz = -0.2
            # model_points[:, 2] += dz

            processed_points = preprocess_raw_for_second(model_points, **PREPROCESS_CFG)

            log_cloud_stats("RAW", raw_points)
            log_cloud_stats("MODEL", model_points)
            log_cloud_stats("PROCESSED", processed_points)

            if processed_points.shape[0] == 0:
                viewer.update(
                    processed_points,
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

            # viewer.update(processed_points, boxes, labels)
            viewer.update(model_points, boxes, labels)
            
            if ENABLE_DEBUG_LOGS:
                print(
                    f"raw={len(raw_points)} | processed={len(processed_points)} | detections={len(boxes)}",
                    end="\r",
                )

            sleep(LOOP_SLEEP_SEC)

    except KeyboardInterrupt:
        print("\nChiusura...")

    finally:
        lidar.stop()
        viewer.close()


if __name__ == "__main__":
    main()