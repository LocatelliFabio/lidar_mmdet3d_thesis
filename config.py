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
    "host_address": "0.0.0.0",
    "group_address": "0.0.0.0",
}

VIEWER_CFG = {
    "window_name": "RS128 Live Detection",
    "width": 1280,
    "height": 720,
    "point_size": 3.0,
}

LOOP_SLEEP_SEC = 0.01
ENABLE_DEBUG_LOGS = False

TRACKER_CFG = {
    "score_thr": 0.40,
    "smoothing_window": 5,
    "max_reasonable_speed_kmh": 50.0,
    "max_match_distance_m": 1.5,
    "max_missed_frames": 3,
    "prefer_high_score_when_unlocked": True,
    "match_distance_margin_m": 0.30,
    "match_distance_scale": 1.25,
}