from time import sleep
from threading import Lock, Thread
import numpy as np
import open3d as o3d

from rs_driver import (
    RSDriverParam,
    InputType,
    LidarType,
    LidarDriver,
    PointCloud,
    Error,
)

from mmdet3d.apis import init_model, inference_detector
from preprocessing.pre_process import preprocess_raw_for_second


# =========================
# CONFIG MODELLO
# =========================
CONFIG = 'mmdet3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py'
CHECKPOINT = 'mmdet3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth'
DEVICE = 'cuda:0'

CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']
CYCLIST_LABEL = 1
SCORE_THR = 0.35


# =========================
# SHARED STATE
# =========================
latest_raw_points = None        # ultimo frame XYZI ricevuto dal lidar
latest_vis_points = None        # punti da mostrare
latest_cyclist_box = None       # box migliore [x,y,z,dx,dy,dz,yaw]
latest_cyclist_score = None

raw_lock = Lock()
result_lock = Lock()

running = True


# =========================
# CALLBACK LIDAR
# =========================
def get_point_cloud_callback() -> PointCloud:
    return PointCloud()


def return_point_cloud_callback(point_cloud: PointCloud):
    global latest_raw_points

    arr = point_cloud.numpy()
    if arr is None or arr.shape[0] == 0:
        return

    # mantieni XYZI se disponibile
    # atteso: arr shape (N, >=4)
    if arr.shape[1] >= 4:
        pts_xyzi = arr[:, :4].astype(np.float32, copy=True)
    else:
        # fallback: crea intensità nulla se non presente
        xyz = arr[:, :3].astype(np.float32, copy=True)
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts_xyzi = np.hstack([xyz, intensity])

    with raw_lock:
        latest_raw_points = pts_xyzi


def exception_callback(error: Error):
    if "OVERFLOW" in str(error):
        print(error)


# =========================
# UTILS BBOX
# =========================
def make_bbox_lineset(box7d):
    """
    box7d = [x, y, z, dx, dy, dz, yaw]
    Restituisce un LineSet Open3D orientato.
    """
    x, y, z, dx, dy, dz, yaw = box7d

    # In molti detector 3D di MMDet3D la z del box LiDAR è sul fondo.
    # Per disegnare il box correttamente in Open3D portiamo il centro al centro geometrico.
    cz = z + dz / 2.0

    # 8 corner nel frame locale del box
    corners = np.array([
        [ dx/2,  dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
        [ dx/2,  dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
    ], dtype=np.float64)

    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    corners = (corners @ R.T) + np.array([x, y, cz], dtype=np.float64)

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],   # top
        [4, 5], [5, 6], [6, 7], [7, 4],   # bottom
        [0, 4], [1, 5], [2, 6], [3, 7]    # vertical
    ]

    colors = [[1.0, 0.0, 0.0] for _ in lines]  # rosso

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def extract_best_cyclist(result, score_thr=0.35):
    """
    Restituisce la bbox del Cyclist con score più alto sopra soglia.
    """
    pred = result.pred_instances_3d

    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()

    mask = (labels == CYCLIST_LABEL) & (scores >= score_thr)
    if not np.any(mask):
        return None, None

    cyclist_scores = scores[mask]
    cyclist_boxes = boxes[mask]

    best_idx = np.argmax(cyclist_scores)
    return cyclist_boxes[best_idx], float(cyclist_scores[best_idx])


# =========================
# THREAD INFERENZA
# =========================
def inference_loop(model):
    global latest_vis_points, latest_cyclist_box, latest_cyclist_score, running

    last_processed_id = -1
    frame_counter = 0

    while running:
        with raw_lock:
            if latest_raw_points is None:
                frame = None
            else:
                frame = latest_raw_points.copy()

        if frame is None:
            sleep(0.005)
            continue

        frame_counter += 1

        try:
            # preprocess per SECOND
            processed_points = preprocess_raw_for_second(frame)

            # inferenza
            result, _ = inference_detector(model, processed_points)

            # prendi il miglior cyclist
            best_box, best_score = extract_best_cyclist(result, score_thr=SCORE_THR)

            # punti da visualizzare:
            # usa il frame raw, ma solo xyz e un leggero sottocampionamento
            vis_xyz = frame[:, :3]
            vis_xyz = vis_xyz[::4]

            with result_lock:
                latest_vis_points = vis_xyz
                latest_cyclist_box = best_box
                latest_cyclist_score = best_score

        except Exception as e:
            print(f"[Inference error] {e}")

        sleep(0.001)


# =========================
# MAIN
# =========================
def main():
    global running

    print("Caricamento modello...")
    model = init_model(CONFIG, CHECKPOINT, device=DEVICE)
    print("Modello caricato.")

    param = RSDriverParam()
    param.lidar_type = LidarType.RS128
    param.input_type = InputType.ONLINE_LIDAR

    param.input_param.msop_port = 6699
    param.input_param.difop_port = 7788
    param.input_param.host_address = "0.0.0.0"
    param.input_param.group_address = "0.0.0.0"

    driver = LidarDriver()
    driver.register_point_cloud_callback(
        get_point_cloud_callback,
        return_point_cloud_callback
    )
    driver.register_exception_callback(exception_callback)

    if not driver.init(param):
        print("Driver init failed")
        return

    driver.start()

    # thread inferenza
    worker = Thread(target=inference_loop, args=(model,), daemon=True)
    worker.start()

    # visualizzazione
    vis = o3d.visualization.Visualizer()
    vis.create_window("RS128 Cyclist Detection", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    bbox_lines = o3d.geometry.LineSet()
    vis.add_geometry(bbox_lines)

    render = vis.get_render_option()
    render.point_size = 2.0
    render.background_color = np.array([0.0, 0.0, 0.0])

    first_frame = True
    last_box_visible = False

    try:
        while True:
            with result_lock:
                pts = None if latest_vis_points is None else latest_vis_points.copy()
                box = None if latest_cyclist_box is None else latest_cyclist_box.copy()
                score = latest_cyclist_score

            if pts is not None and len(pts) > 0:
                pcd.points = o3d.utility.Vector3dVector(pts)
                vis.update_geometry(pcd)

                if first_frame:
                    vis.reset_view_point(True)
                    first_frame = False

            # aggiorna bbox
            if box is not None:
                new_bbox = make_bbox_lineset(box)
                bbox_lines.points = new_bbox.points
                bbox_lines.lines = new_bbox.lines
                bbox_lines.colors = new_bbox.colors
                vis.update_geometry(bbox_lines)

                if score is not None:
                    print(f"Cyclist score: {score:.3f}", end="\r")

                last_box_visible = True

            else:
                # svuota la bbox se nel frame corrente non c'è cyclist
                if last_box_visible:
                    bbox_lines.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    bbox_lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                    bbox_lines.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    vis.update_geometry(bbox_lines)
                    last_box_visible = False

            vis.poll_events()
            vis.update_renderer()
            sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        running = False
        worker.join(timeout=1.0)
        driver.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()