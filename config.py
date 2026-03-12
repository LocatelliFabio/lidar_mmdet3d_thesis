# config.py

MODEL_CONFIG = "mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py"
MODEL_CHECKPOINT = "mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth"
MODEL_DEVICE = "cuda:0"

CLASS_NAMES = ["Pedestrian", "Cyclist", "Car"]

SCORE_THR = 0.30

PC_RANGE = (0, -40, -3, 70.4, 40, 1)

PREPROCESS_CFG = {
    "pc_range": PC_RANGE,
    "ds_voxel": 0.08,
    "denoise": False,
    "den_voxel": 0.35,
    "den_min_pts": 2,
    "ground_cell": 0.5,
    "ground_thresh": 0.07,
}

LIDAR_CFG = {
    "msop_port": 6699,
    "difop_port": 7788,
    #"host_address": "192.168.1.102",
    "host_address": "127.0.0.1",
    "group_address": "0.0.0.0",
}

VIEWER_CFG = {
    "window_name": "RS128 Live Detection",
    "width": 1280,
    "height": 720,
    "point_size": 1.5,

    "camera_position": [-1.0, 0.0, 0.0],
    "camera_lookat": [1.0, 0.0, 0.0],
}

LOOP_SLEEP_SEC = 0.002
ENABLE_DEBUG_LOGS = False

SPEED_CFG = {
    "score_thr": SCORE_THR,
    "smoothing_window": 5,
    "use_xy_only": True,
    "max_reasonable_speed_kmh": 80.0,
}