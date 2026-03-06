import os
import sys
import time
import threading
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("bridge/build"))

from rs_bridge import RsBridge
from preprocessing.pre_process import preprocess_raw_for_second
from mmdet3d.apis import inference_detector
from model_factory import get_model


CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']
LABEL_COLORS = {
    0: (0, 1, 0),  # Pedestrian
    1: (0, 0, 1),  # Cyclist
    2: (1, 0, 0),  # Car
}


# ============================================================
# Utility Open3D
# ============================================================

def _colors_from_intensity(intensity: np.ndarray) -> np.ndarray:
    intensity = intensity.astype(np.float32, copy=False)
    ptp = float(np.ptp(intensity))
    if ptp < 1e-6:
        intensity_norm = np.zeros_like(intensity, dtype=np.float32)
    else:
        intensity_norm = (intensity - intensity.min()) / (ptp + 1e-6)
    return plt.get_cmap("viridis")(intensity_norm)[:, :3]


def _make_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    xyz = points[:, :3]
    intensity = points[:, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(
        _colors_from_intensity(intensity).astype(np.float64, copy=False)
    )
    return pcd


def _update_pcd_geometry(pcd: o3d.geometry.PointCloud, points: np.ndarray):
    xyz = points[:, :3]
    intensity = points[:, 3]

    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(
        _colors_from_intensity(intensity).astype(np.float64, copy=False)
    )


def _boxes_from_result(result, score_thr: float):
    pred = result.pred_instances_3d
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()

    keep = scores >= score_thr
    return boxes[keep], scores[keep], labels[keep]


def _make_obb(box: np.ndarray, label: int) -> o3d.geometry.OrientedBoundingBox:
    center = box[:3].copy()
    dims = box[3:6].copy()
    yaw = float(box[6])

    # Conversione yaw per Open3D
    yaw = -yaw + np.pi / 2

    # MMDet3D: bottom-center -> Open3D: center
    center[2] += dims[2] / 2.0

    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, yaw])
    obb = o3d.geometry.OrientedBoundingBox(center, R, dims)
    obb.color = LABEL_COLORS.get(int(label), (1, 1, 1))
    return obb


def _obb_to_lineset(obb: o3d.geometry.OrientedBoundingBox, color):
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color(color)
    return ls


# ============================================================
# Thread acquisizione LiDAR
# ============================================================

class LidarFrameBuffer:
    """
    Tiene sempre solo l'ultimo frame disponibile.
    Se l'inferenza è lenta, i frame vecchi vengono scartati:
    ottimo per visualizzazione real-time.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_points = None
        self.running = False
        self.thread = None

    def start(self, bridge: RsBridge, timeout_ms=200):
        self.running = True

        def _worker():
            while self.running:
                try:
                    pts = bridge.get_frame(timeout_ms=timeout_ms)
                    if pts is not None:
                        with self.lock:
                            self.latest_points = pts
                except Exception as e:
                    print(f"[ACQ] errore acquisizione: {e}")
                    time.sleep(0.01)

        self.thread = threading.Thread(target=_worker, daemon=True)
        self.thread.start()

    def get_latest(self):
        with self.lock:
            pts = self.latest_points
            self.latest_points = None
        return pts

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


# ============================================================
# Visualizzazione real-time
# ============================================================

def visualize_realtime(
    bridge: RsBridge,
    model,
    score_thr: float = 0.2,
    warmup_frames: int = 5,
):
    # Warmup iniziale sensore
    print(f"[INFO] warmup frames: {warmup_frames}")
    for i in range(warmup_frames):
        _ = bridge.get_frame(timeout_ms=3000)
        print(f"[INFO] warmup {i+1}/{warmup_frames}")

    # Buffer thread-safe che tiene l'ultimo frame
    buffer = LidarFrameBuffer()
    buffer.start(bridge, timeout_ms=200)

    # Aspetta il primo frame valido
    print("[INFO] attendo primo frame...")
    raw_points = None
    while raw_points is None:
        raw_points = buffer.get_latest()
        time.sleep(0.01)

    proc_points = preprocess_raw_for_second(raw_points)

    # Prima geometria
    pcd = _make_pcd(proc_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Real-Time", width=1280, height=800)
    vis.add_geometry(pcd)

    render_opt = vis.get_render_option()
    render_opt.point_size = 1.5
    render_opt.background_color = np.array([0.0, 0.0, 0.0])

    ctr = vis.get_view_control()
    ctr.set_front(np.array([-1.0, 0.0, 0.4]))
    ctr.set_lookat(np.array([10.0, 0.0, 0.0]))
    ctr.set_up(np.array([0.0, 0.0, 1.0]))
    ctr.set_zoom(0.08)

    current_box_geometries = []

    frame_count = 0
    last_time = time.time()

    try:
        while True:
            raw_points = buffer.get_latest()

            # Se non c'è ancora un frame nuovo, mantieni la finestra viva
            alive = vis.poll_events()
            vis.update_renderer()
            if alive is False:
                break

            if raw_points is None:
                time.sleep(0.005)
                continue

            # Preprocess
            proc_points = preprocess_raw_for_second(raw_points)

            # Inferenza
            result, _ = inference_detector(model, proc_points)

            # Aggiorna point cloud
            _update_pcd_geometry(pcd, proc_points)
            vis.update_geometry(pcd)

            # Rimuovi box precedenti
            for geom in current_box_geometries:
                vis.remove_geometry(geom, reset_bounding_box=False)
            current_box_geometries.clear()

            # Crea nuove box
            boxes, scores, labels = _boxes_from_result(result, score_thr)
            for box, score, label in zip(boxes, scores, labels):
                obb = _make_obb(box, int(label))
                color = LABEL_COLORS.get(int(label), (1, 1, 1))
                ls = _obb_to_lineset(obb, color)
                vis.add_geometry(ls, reset_bounding_box=False)
                current_box_geometries.append(ls)

            # Render
            alive = vis.poll_events()
            vis.update_renderer()
            if alive is False:
                break

            # FPS debug
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                print(f"[INFO] FPS visualizzazione/inferenza: {frame_count}")
                frame_count = 0
                last_time = now

    except KeyboardInterrupt:
        print("[INFO] interrotto da tastiera")

    finally:
        buffer.stop()
        vis.destroy_window()


# ============================================================
# Main
# ============================================================

def main():
    bridge = RsBridge()
    bridge.configure(
        host_ip="192.168.1.102",
        msop_port=6699,
        difop_port=7788,
        use_rsp128=False,
        split_angle_deg=0.0
    )
    bridge.start()

    print("[INFO] caricamento modello...")
    model = get_model()
    print("[INFO] modello caricato")

    visualize_realtime(
        bridge=bridge,
        model=model,
        score_thr=0.2,
        warmup_frames=0,
    )


if __name__ == "__main__":
    main()