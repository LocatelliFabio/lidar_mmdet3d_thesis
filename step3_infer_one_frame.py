from __future__ import annotations
import numpy as np

from mmdet3d.apis import init_model, inference_detector

from pcd_reader import read_pcd_xyzi_ascii

# ==========================
PCD_PATH = r"pcd_tests/test07/test07 (Frame 0076).pcd"
#PCD_PATH = r"pcd_tests/pcd_001.pcd"
CONFIG   = r"mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py"
CKPT     = r"mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth"
DEVICE   = "cuda:0"   # oppure "cpu"
# ==========================

def main():
    pts = read_pcd_xyzi_ascii(PCD_PATH).astype(np.float32, copy=False)
    print("Loaded points:", pts.shape, pts.dtype)

    model = init_model(CONFIG, CKPT, device=DEVICE)
    result = inference_detector(model, pts)

    # print("Result: ", result)

    # MMDet3D 1.1.1: result di solito è Det3DDataSample
    pred = result[0].pred_instances_3d
    bboxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()

    print("Detections:", bboxes.shape[0])
    if bboxes.shape[0] == 0:
        return

    # ordina per score decrescente
    order = np.argsort(-scores)
    bboxes, scores, labels = bboxes[order], scores[order], labels[order]

    print("\nTop 10:")
    for i in range(min(10, bboxes.shape[0])):
        x, y, z, dx, dy, dz, yaw = bboxes[i].tolist()
        print(
            f"{i:02d}  score={scores[i]:.3f}  label={int(labels[i])}  "
            f"box=[x={x:.2f}, y={y:.2f}, z={z:.2f}, dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, yaw={yaw:.2f}]"
        )

if __name__ == "__main__":
    main()