from time import sleep
from threading import Lock, Thread
import numpy as np

from rs_driver import (
    RSDriverParam,
    InputType,
    LidarType,
    LidarDriver,
    PointCloud,
    Error,
)

from versione1.detector_live import LiveDetector
from versione1.viewer_live import LiveViewer


# =========================
# CONFIG
# =========================
CONFIG = 'mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py'
CHECKPOINT = 'mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth'
DEVICE = 'cuda:0'
SCORE_THR = 0.35

RAW_VIS_DOWNSAMPLE = 4   # 1=tutti, 2=uno ogni 2, 4=uno ogni 4
AXIS_SIZE = 2.0


# =========================
# SHARED STATE
# =========================
latest_raw_points = None
latest_detections = []

raw_lock = Lock()
result_lock = Lock()

running = True


# =========================
# CALLBACK LIDAR
# =========================
def get_point_cloud_callback() -> PointCloud:
    return PointCloud()


def return_point_cloud_callback(point_cloud: PointCloud):
    global latest_raw_points

    arr = point_cloud.numpy()
    if arr is None or arr.shape[0] == 0:
        return
    
    arr[:, :2] *= -1

    if arr.shape[1] >= 4:
        pts_xyzi = arr[:, :4].astype(np.float32, copy=True)
    else:
        xyz = arr[:, :3].astype(np.float32, copy=True)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts_xyzi = np.hstack([xyz, intensity])

    with raw_lock:
        latest_raw_points = pts_xyzi


def exception_callback(error: Error):
    if "OVERFLOW" in str(error):
        print(error)


# =========================
# THREAD INFERENZA
# =========================
def inference_loop(detector: LiveDetector):
    global running
    global latest_detections

    while running:
        with raw_lock:
            frame = None if latest_raw_points is None else latest_raw_points.copy()

        if frame is None:
            sleep(0.005)
            continue

        try:
            processed = detector.preprocess(frame)
            result = detector.infer(processed)
            detections = detector.extract_boxes(result)

            with result_lock:
                latest_detections = detections

        except Exception as e:
            print(f"[Inference error] {e}")

        sleep(0.001)


def print_detection_summary(detections):
    counts = {0: 0, 1: 0, 2: 0}

    for det in detections:
        label = det['label']
        if label in counts:
            counts[label] += 1

    text = (
        f"Pedestrian: {counts[0]} | "
        f"Cyclist: {counts[1]} | "
        f"Car: {counts[2]}"
    )
    print(text, end="\r")


# =========================
# MAIN
# =========================
def main():
    global running

    print("Caricamento modello...")
    detector = LiveDetector(
        config=CONFIG,
        checkpoint=CHECKPOINT,
        device=DEVICE,
        score_thr=SCORE_THR
    )
    detector.load()
    print("Modello caricato.")

    param = RSDriverParam()
    param.lidar_type = LidarType.RS128
    param.input_type = InputType.ONLINE_LIDAR

    param.input_param.msop_port = 6699
    param.input_param.difop_port = 7788
    param.input_param.host_address = "0.0.0.0"
    param.input_param.group_address = "0.0.0.0"

    driver = LidarDriver()
    driver.register_point_cloud_callback(
        get_point_cloud_callback,
        return_point_cloud_callback
    )
    driver.register_exception_callback(exception_callback)

    if not driver.init(param):
        print("Driver init failed")
        return

    driver.start()

    worker = Thread(target=inference_loop, args=(detector,), daemon=True)
    worker.start()

    viewer = LiveViewer(
        width=1280,
        height=720,
        title="RS128 Real-Time Detection",
        axis_size=AXIS_SIZE
    )
    viewer.create()

    try:
        while True:
            with raw_lock:
                raw_frame = None if latest_raw_points is None else latest_raw_points.copy()

            with result_lock:
                detections = list(latest_detections)

            if raw_frame is not None and len(raw_frame) > 0:
                raw_points_xyz = raw_frame[:, :3]

                if RAW_VIS_DOWNSAMPLE > 1:
                    raw_points_xyz = raw_points_xyz[::RAW_VIS_DOWNSAMPLE]

                viewer.update(raw_points_xyz, detections)
                print_detection_summary(detections)
            else:
                viewer.update(np.zeros((0, 3), dtype=np.float64), [])

            sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        running = False
        worker.join(timeout=1.0)
        driver.stop()
        viewer.destroy()


if __name__ == "__main__":
    main()