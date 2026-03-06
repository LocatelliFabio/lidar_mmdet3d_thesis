from __future__ import annotations
import numpy as np

from mmdet3d.apis import init_model, inference_detector
from pcd_reader import read_pcd_xyzi_ascii

import open3d as o3d

# ====== MODIFICA QUI ======
PCD_PATH = r"pcd_tests/pcd_007.pcd"
CONFIG   = r"mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py"
CKPT     = r"mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth"
DEVICE   = "cuda:0"   # oppure "cpu"
SCORE_THR = 0.30
# ==========================

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]

def yaw_to_Rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def make_o3d_obb(x, y, z, dx, dy, dz, yaw):
    center = np.array([x, y, z], dtype=np.float64)
    R = yaw_to_Rz(yaw)
    extent = np.array([dx, dy, dz], dtype=np.float64)

    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    return ls

def main():
    pts = read_pcd_xyzi_ascii(PCD_PATH).astype(np.float32, copy=False)
    print("Loaded points:", pts.shape, pts.dtype)

    model = init_model(CONFIG, CKPT, device=DEVICE)
    result = inference_detector(model, pts)

    pred = result[0].pred_instances_3d
    bboxes = pred.bboxes_3d.tensor.detach().cpu().numpy()   # (N,7) [x,y,z,dx,dy,dz,yaw]
    scores = pred.scores_3d.detach().cpu().numpy()          # (N,)
    labels = pred.labels_3d.detach().cpu().numpy()          # (N,)

    # filtro per soglia
    keep = scores >= SCORE_THR
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

    print("Detections after thr:", bboxes.shape[0])
    if bboxes.shape[0] == 0:
        # visualizza solo la point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        o3d.visualization.draw_geometries([pcd])
        return

    # ordina per score
    order = np.argsort(-scores)
    bboxes, scores, labels = bboxes[order], scores[order], labels[order]

    # --- Open3D geometries ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))

    # (opzionale) colora la point cloud usando intensity (pts[:,3])
    if pts.shape[1] >= 4:
        inten = pts[:, 3]
        inten = (inten - inten.min()) / (inten.ptp() + 1e-9)
        colors = np.stack([inten, inten, inten], axis=1).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # colori box per classe (Car=rosso, Ped=verde, Cyclist=blu)
    class_colors = {
        0: [1.0, 0.2, 0.2],
        1: [0.2, 1.0, 0.2],
        2: [0.2, 0.2, 1.0],
    }

    geoms = [pcd]
    for i in range(bboxes.shape[0]):
        x, y, z, dx, dy, dz, yaw = bboxes[i].tolist()
        ls = make_o3d_obb(x, y, z, dx, dy, dz, yaw)
        ls.paint_uniform_color(class_colors.get(int(labels[i]), [1.0, 1.0, 0.0]))
        geoms.append(ls)

        if i < 10:
            print(f"{i:02d} {CLASS_NAMES[int(labels[i])]} score={scores[i]:.3f}")

    # aggiungi assi (utile per orientamento)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geoms.append(axis)

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()