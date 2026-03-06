import glob

from postprocessing.utils import export_bike_pcd
from postprocessing.utils import extract_points_from_bb_sequence
from postprocessing.utils import get_distance_and_speed
from preprocessing.pre_process import preprocess_raw_for_second
from pcd_reader import read_pcd_xyzi_ascii
from mmdet3d.apis import init_model, inference_detector

pcd_files = sorted(glob.glob("pcd_tests/test07/*.pcd"))

config = 'mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py'
checkpoint = 'mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth'
model = init_model(config, checkpoint, device='cuda:0')

results = []
for p in pcd_files:
    raw = read_pcd_xyzi_ascii(p)
    pts = preprocess_raw_for_second(raw)
    result, _ = inference_detector(model, pts)
    results.append(result)

pcd_list, centers, boxes7_norm, scores = extract_points_from_bb_sequence(
    pcd_files, results, score_thr=0.4, model_name="second"
)

covered, speed_arr, vmax, vmean, dist_steps = get_distance_and_speed(pcd_list, fps=10.0)

print("Covered distance (m):", covered)
print("Speed array (km/h):", speed_arr)
print("Max speed (km/h):", vmax)
print("Mean speed (km/h):", vmean)
print("Distance steps (m):", dist_steps)

export_bike_pcd(pcd_list, out_dir="extracted_bike_pcd")