import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt


CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']
LABEL_COLORS = {
    0: (0, 1, 0),  # Pedestrian
    1: (0, 0, 1),  # Cyclist
    2: (1, 0, 0),  # Car
}


def _colors_from_intensity(intensity: np.ndarray) -> np.ndarray:
    ptp = float(intensity.ptp())
    if ptp < 1e-6:
        intensity_norm = np.zeros_like(intensity, dtype=np.float32)
    else:
        intensity_norm = (intensity - intensity.min()) / (ptp + 1e-6)
    return plt.get_cmap("viridis")(intensity_norm)[:, :3]


def _make_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    xyz = points[:, :3]
    intensity = points[:, 3]
    colors = _colors_from_intensity(intensity)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64, copy=False))
    return pcd


def _boxes_from_result(result, score_thr: float):
    pred = result.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()

    keep = scores >= score_thr
    return boxes[keep], scores[keep], labels[keep]


def _make_obb_and_text(box: np.ndarray, score: float, label: int):
    center = box[:3].copy()
    dims = box[3:6].copy()
    yaw = float(box[6])

    # yaw convention found to match Open3D
    yaw = -yaw + np.pi / 2

    # MMDet3D gives bottom-center, Open3D wants center
    center[2] += dims[2] / 2.0

    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

    obb = o3d.geometry.OrientedBoundingBox(center, R, dims)
    obb.color = LABEL_COLORS.get(int(label), (1, 1, 1))

    text_pos = center.copy()
    text_pos[2] += dims[2] / 2 + 0.3

    name = CLASS_NAMES[int(label)] if 0 <= int(label) < len(CLASS_NAMES) else f"cls{int(label)}"
    text = f"{name} {float(score):.2f}"
    return obb, text_pos, text


def visualize_points_and_bboxes(points: np.ndarray, result, score_thr: float = 0.2):
    pcd = _make_pcd(points)
    boxes, scores, labels = _boxes_from_result(result, score_thr)

    geometries = [pcd]
    for box, score, label in zip(boxes, scores, labels):
        obb, _, _ = _make_obb_and_text(box, float(score), int(label))
        geometries.append(obb)

    o3d.visualization.draw_geometries(geometries)


def visualize_bboxes_with_scores(points: np.ndarray, result, score_thr: float = 0.2):
    pcd = _make_pcd(points)
    boxes, scores, labels = _boxes_from_result(result, score_thr)

    obbs = []
    labels_3d = []
    for box, score, label in zip(boxes, scores, labels):
        obb, pos, text = _make_obb_and_text(box, float(score), int(label))
        obbs.append(obb)
        labels_3d.append((pos, text))

    # GUI init (compatibile con versioni senza is_running)
    try:
        gui.Application.instance.initialize()
    except Exception:
        pass

    vis = o3d.visualization.O3DVisualizer("PointCloud + BBoxes", 1024, 768)
    vis.show_settings = True

    vis.add_geometry("pcd", pcd)
    for i, obb in enumerate(obbs):
        vis.add_geometry(f"box_{i}", obb)

    for pos, text in labels_3d:
        vis.add_3d_label(pos, text)

    vis.reset_camera_to_default()
    
    vis.setup_camera(
        60.0,
        [20, 0, 0],
        [-30, 0, 20],
        [0, 0, 1]
    )

    gui.Application.instance.add_window(vis)
    gui.Application.instance.run()