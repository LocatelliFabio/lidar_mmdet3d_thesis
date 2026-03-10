# basic_inference.py

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

#vl = model.cfg.model.data_preprocessor.voxel_layer
#print("voxel_size:", vl.voxel_size)
#print("max_num_points:", vl.max_num_points)
#print("max_voxels:", vl.max_voxels)

#pc_range = model.cfg.model.data_preprocessor.voxel_layer.point_cloud_range

pcd_path = "data/test10/0_original/test10 (Frame 0044).pcd"
#pcd_path = "pcd_tests/test07_1_pcd/pcd_007.pcd"

raw_points = read_pcd_xyzi_ascii(pcd_path)   # (N,4) float32
#print(len(points))
#print(points[:5])

print_min_max(raw_points)

#points = crop_point_cloud(points)
#points[:, 3] = np.clip(points[:, 3], 0, 255) / 255.0

# timed_pipeline(
#     raw_points,
#     model,
#     preprocess_fn=lambda x: preprocess_raw_for_second(
#         x,
#         ds_voxel=0.10,
#         denoise=False,
#         ground_cell=0.5,
#         ground_thresh=0.05
#     ),
#     runs=20
# )
# exit()

points = preprocess_raw_for_second(raw_points)

print("Punti raw:", len(raw_points))
print("Punti dopo pre-process:", len(points))

result, data = inference_detector(model, points)

pred = result.pred_instances_3d
scores = pred.scores_3d.detach().cpu().numpy()
labels = pred.labels_3d.detach().cpu().numpy()
boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()

class_names = ['Pedestrian', 'Cyclist', 'Car']

for cls_id, name in [(0,"Pedestrian"), (1,"Cyclist"), (2,"Car")]:
    cls_scores = scores[labels == cls_id]
    print(name, "count:", cls_scores.size,
          "max:", (cls_scores.max() if cls_scores.size else None),
          "top5:", np.sort(cls_scores)[-5:] if cls_scores.size else None)
    

#print(result)
#print(points)
#preview_yaw_conventions(points, result, score_thr=0.2)
visualize_bboxes_with_scores(raw_points, result)

# show_points_grid(raw_points)
#visualize_points_and_bboxes(points, result)