# basic_inference_all_pcd.py

import os
import glob
import numpy as np

from mmdet3d.apis import init_model, inference_detector
from pcd_reader import read_pcd_xyzi_ascii
from show_pcd import visualize_points_and_bboxes, visualize_bboxes_with_scores
from yaw_variants import preview_yaw_conventions
from preprocessing.pre_process import preprocess_raw_for_second
from timedpipeline import timed_pipeline
from show_points2 import show_points_grid
from print_min_max_nparray import print_min_max

config = 'mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py'
checkpoint = 'mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth'

model = init_model(config, checkpoint, device='cuda:0')

# Cartella contenente i file .pcd
pcd_dir = "data/test10/0_original"

# Cerca tutti i file .pcd nella cartella
pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

if not pcd_files:
    raise FileNotFoundError(f"Nessun file .pcd trovato in: {pcd_dir}")

class_names = ['Pedestrian', 'Cyclist', 'Car']
best_dz = -1   # da trovare sperimentalmente

for pcd_path in pcd_files:
    print("\n" + "=" * 80)
    print(f"Processing: {pcd_path}")
    print("=" * 80)

    raw_points = read_pcd_xyzi_ascii(pcd_path)

    # niente dz per ora
    # raw_points[:, 2] += dz


    points = preprocess_raw_for_second(raw_points)
    result, data = inference_detector(model, points)

    pred = result.pred_instances_3d
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()

    for cls_id, name in [(0, "Pedestrian"), (1, "Cyclist"), (2, "Car")]:
        cls_scores = scores[labels == cls_id]
        print(
            name,
            "count:", cls_scores.size,
            "max:", (cls_scores.max() if cls_scores.size else None),
            "top5:", np.sort(cls_scores)[-5:] if cls_scores.size else None
        )

    # Visualizzazione
    visualize_bboxes_with_scores(points, result)

    # Alternative opzionali:
    # preview_yaw_conventions(points, result, score_thr=0.2)
    # visualize_points_and_bboxes(points, result)
    # show_points_grid(raw_points)