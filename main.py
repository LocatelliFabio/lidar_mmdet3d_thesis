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
)

from thread_safe_buffer import LatestValueBuffer
from rs_lidar_stream import RSLidarStream
from preprocessing.pre_process import preprocess_raw_for_second
from detector import SecondDetector
from live_viewer import LiveViewer3D


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

            # test_points = raw_points.copy()
            # test_points[:, 0] = raw_points[:, 1]
            # test_points[:, 1] = -raw_points[:, 0]
            # test_points[:, 2] = raw_points[:, 2]

            # raw_points = test_points

            processed_points = preprocess_raw_for_second(raw_points, **PREPROCESS_CFG)

            # print(processed_points[:5])

            if len(processed_points) == 0:
                viewer.update(processed_points, np.empty((0, 7), dtype=np.float32))
                sleep(0.01)
                continue

            result = detector.infer(processed_points)
            boxes, scores, labels = detector.extract_predictions(result, score_thr=SCORE_THR)

            viewer.update(raw_points, boxes)

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