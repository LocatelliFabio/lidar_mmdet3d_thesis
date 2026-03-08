from time import sleep
import numpy as np

from config import (
    MODEL_CONFIG,
    MODEL_CHECKPOINT,
    MODEL_DEVICE,
    PREPROCESS_CFG,
    SCORE_THR,
    LIDAR_CFG,
)

from thread_safe_buffer import LatestValueBuffer
from rs_lidar_stream import RSLidarStream
from preprocessing.pre_process import preprocess_raw_for_second
from detector import SecondDetector
from live_viewer import LiveViewer3D


def rs_to_model_coords(points: np.ndarray) -> np.ndarray:
    # print(points[:5])
    out = points.copy()
    out[:, 0] = points[:, 1]
    out[:, 1] = points[:, 2]
    out[:, 2] = points[:, 0]
    if points.shape[1] > 3:
        out[:, 3] = points[:, 3]
    return out


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

    viewer = LiveViewer3D(window_name="RS128 Live Detection")

    lidar.start()

    try:
        while True:
            raw_points = frame_buffer.get_copy()

            if raw_points is None or len(raw_points) == 0:
                viewer.spin_once()
                sleep(0.01)
                continue

            model_points = rs_to_model_coords(raw_points)

            processed_points = preprocess_raw_for_second(model_points, **PREPROCESS_CFG)

            if len(processed_points) == 0:
                viewer.update(processed_points,
                              np.empty((0, 7), dtype=np.float32),
                              np.empty((0,), dtype=np.int64))
                sleep(0.01)
                continue

            result = detector.infer(processed_points)
            boxes, scores, labels = detector.extract_predictions(result, score_thr=SCORE_THR)

            viewer.update(processed_points, boxes, labels)

            print(
                f"raw={len(raw_points)} | processed={len(processed_points)} | det={len(boxes)}",
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