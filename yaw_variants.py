import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt


def _init_gui():
    try:
        gui.Application.instance.initialize()
    except Exception:
        pass


def _make_pcd(points):
    xyz = points[:, :3]
    intensity = points[:, 3]

    intensity_norm = (intensity - intensity.min()) / (intensity.ptp() + 1e-6)
    colors = plt.get_cmap("viridis")(intensity_norm)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))

    return pcd


def preview_yaw_conventions(points: np.ndarray, result, score_thr=0.2):

    class_names = ['Pedestrian', 'Cyclist', 'Car']

    label_colors = {
        0: (0, 1, 0),
        1: (0, 0, 1),
        2: (1, 0, 0)
    }

    pred = result.pred_instances_3d

    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()

    keep = scores >= score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(boxes) == 0:
        print("Nessuna box sopra soglia")
        return

    variants = [
        ("yaw", lambda y: y),
        ("-yaw", lambda y: -y),
        ("yaw+pi/2", lambda y: y + np.pi / 2),
        ("-yaw+pi/2", lambda y: -y + np.pi / 2),
    ]

    pcd = _make_pcd(points)
    _init_gui()

    for name, yaw_fn in variants:

        vis = o3d.visualization.O3DVisualizer(f"Yaw convention: {name}", 1024, 768)
        vis.show_settings = True

        vis.add_geometry("pcd", pcd)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):

            center = box[:3].copy()
            dims = box[3:6].copy()

            # Open3D vuole il CENTRO della box.
            # Se MMDet3D ti dà bottom-center (molto comune in KITTI),
            # rialza di metà altezza:
            center[2] += dims[2] / 2.0

            yaw = yaw_fn(float(box[6]))

            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

            obb = o3d.geometry.OrientedBoundingBox(center, R, dims)
            obb.color = label_colors.get(int(label), (1, 1, 1))

            vis.add_geometry(f"box_{i}", obb)

            text_pos = center.copy()
            text_pos[2] += dims[2] / 2 + 0.3

            label_text = f"{class_names[int(label)]} {score:.2f}"

            vis.add_3d_label(text_pos, label_text)

        vis.reset_camera_to_default()

        gui.Application.instance.add_window(vis)
        gui.Application.instance.run()