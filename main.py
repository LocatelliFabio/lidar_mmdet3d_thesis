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
)

from thread_safe_buffer import LatestValueBuffer
from rs_lidar_stream import RSLidarStream
from preprocessing.pre_process import preprocess_raw_for_second
from detector import SecondDetector
from live_viewer import LiveViewer3D


def rs_to_model_coords(points: np.ndarray) -> np.ndarray:
    return points.copy()


def main():
    frame_buffer = LatestValueBuffer()

    lidar = RSLidarStream(frame_buffer=frame_buffer, lidar_cfg=LIDAR_CFG)
    if not lidar.init():
        print("Driver init failed")
        return

    detector = SecondDetector(
        config_path=MODEL_CONFIG,
        checkpoint_path=MODEL_CHECKPOINT,
        device=MODEL_DEVICE
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

            if raw_points is None or len(raw_points) == 0:
                viewer.spin_once()
                sleep(0.01)
                continue

            raw_finite = raw_points[np.isfinite(raw_points[:, :3]).all(axis=1)]

            print("RAW min xyz:", raw_finite[:, :3].min(axis=0))
            print("RAW max xyz:", raw_finite[:, :3].max(axis=0))

            model_points = rs_to_model_coords(raw_finite)

            print("MODEL min xyz:", model_points[:, :3].min(axis=0))
            print("MODEL max xyz:", model_points[:, :3].max(axis=0))

            processed_points = preprocess_raw_for_second(model_points, **PREPROCESS_CFG)

            if len(processed_points) > 0:
                print("processed min xyz:", processed_points[:, :3].min(axis=0))
                print("processed max xyz:", processed_points[:, :3].max(axis=0))
                print("processed shape:", processed_points.shape)

            if len(processed_points) == 0:
                viewer.update(
                    processed_points,
                    np.empty((0, 7), dtype=np.float32),
                    np.empty((0,), dtype=np.int64)
                )
                sleep(0.01)
                continue

            result = detector.infer(processed_points)
            boxes, scores, labels = detector.extract_predictions(
                result,
                score_thr=SCORE_THR
            )

            # print("processed shape:", processed_points.shape)
            # print("processed min xyz:", processed_points[:, :3].min(axis=0))
            # print("processed max xyz:", processed_points[:, :3].max(axis=0))

            viewer.update(processed_points, boxes, labels)

            print(
                f"raw={len(raw_points)} | processed={len(processed_points)} | detections={len(boxes)}",
                end="\r"
            )

            sleep(0.01)

    except KeyboardInterrupt:
        print("\nChiusura...")

    finally:
        lidar.stop()
        viewer.close()


if __name__ == "__main__":
    main()