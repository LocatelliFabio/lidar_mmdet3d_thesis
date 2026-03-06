import numpy as np
import open3d as o3d
import os
from sklearn.cluster import KMeans
from pcd_reader import read_pcd_xyzi_ascii
from preprocessing.pre_process import preprocess_raw_for_second

def points_in_oriented_box(points_xyz: np.ndarray, box7: np.ndarray,
                           pad=(0.25, 0.25, 0.5)):
    """
    points_xyz: (N,3)
    box7: [cx,cy,cz,dx,dy,dz,yaw]
    pad: margine extra (px,py,pz) come nel MATLAB
    return: mask boolean (N,)
    """
    cx, cy, cz, dx, dy, dz, yaw = box7.astype(np.float64)
    C = np.array([cx, cy, cz], dtype=np.float64)

    half = np.array([dx, dy, dz], dtype=np.float64) / 2.0
    half = half + np.array(pad, dtype=np.float64)

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[ c, -s, 0],
                  [ s,  c, 0],
                  [ 0,  0, 1]], dtype=np.float64)

    # porta i punti nel frame locale box: Plocal = R^T (P - C)
    Plocal = (points_xyz - C) @ R   # (N,3)  (equivalente a R^T se usi @R con R definita come sopra)
    # Se vuoi essere ultra-fedele: Plocal = (R.T @ (points_xyz.T - C[:,None])).T

    mask = (np.abs(Plocal[:, 0]) <= half[0]) & \
           (np.abs(Plocal[:, 1]) <= half[1]) & \
           (np.abs(Plocal[:, 2]) <= half[2])
    return mask

def normalize_yaw_over_time(boxes7_list):
    """
    boxes7_list: lista di (7,) oppure None per frame senza detection
    normalizza yaw per evitare flip di 180° tra frame consecutivi (come MATLAB).
    """
    yaws = []
    for b in boxes7_list:
        if b is None:
            yaws.append(None)
        else:
            yaws.append(np.degrees(b[6]))

    yaw_prev = None
    out = []
    for i, yd in enumerate(yaws):
        if yd is None:
            out.append(None); continue
        if yaw_prev is None:
            yaw_prev = yd
            out.append(yd); continue

        diff = yaw_prev - yd
        if diff < -90:
            yd = yd - 180
        elif diff > 90:
            yd = yd + 180

        out.append(yd)
        yaw_prev = yd

    # rimetti in radianti
    boxes7_out = []
    for b, yd in zip(boxes7_list, out):
        if b is None:
            boxes7_out.append(None)
        else:
            b2 = b.copy()
            b2[6] = np.radians(yd)
            boxes7_out.append(b2)
    return boxes7_out


def filter_cyclist_from_mmdet(pred, score_thr=0.4, model_name="second",
                              sensor_height_fix=1.7/2):
    """
    pred: result.pred_instances_3d
    ritorna lista di box7 per frame? -> qui è per un singolo frame.
    Se processi una sequenza, applica frame-by-frame e poi normalize_yaw_over_time.
    """
    scores = pred.scores_3d.detach().cpu().numpy()
    labels = pred.labels_3d.detach().cpu().numpy()
    boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()  # (N,7)

    mask = (labels == 1) & (scores >= score_thr)  # Cyclist
    boxes_c = boxes[mask]
    scores_c = scores[mask]

    if boxes_c.shape[0] == 0:
        return None, None

    # prendi la migliore (come MATLAB prende B(1,:))
    k = int(np.argmax(scores_c))
    b = boxes_c[k].copy()
    s = float(scores_c[k])

    # Fix "second"
    if model_name == "second":
        # cz offset (KITTI sensor height trick)
        b[2] = b[2] + sensor_height_fix

        # swap dx/dy
        dx, dy = b[3], b[4]
        b[3], b[4] = dy, dx

        # invert yaw
        b[6] = -b[6]

    return b, s



def remove_ground_ransac(points_xyzi, distance_thresh=0.05, ransac_n=3, num_iter=200):
    """
    points_xyzi: (N,4) float
    ritorna points_no_ground (M,4)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyzi[:, :3].astype(np.float64))

    if len(pcd.points) < 50:
        return points_xyzi

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_thresh,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iter)
    inliers = np.array(inliers, dtype=np.int64)
    mask_ground = np.zeros(len(points_xyzi), dtype=bool)
    mask_ground[inliers] = True
    return points_xyzi[~mask_ground]


def extract_points_from_bb_sequence(pcd_files, results_per_frame,
                                    score_thr=0.4, model_name="second"):
    T = len(pcd_files)

    raw_points = []
    boxes7_list = [None] * T
    scores_list = [None] * T

    # 1) Leggi punti e filtra bbox
    for i, (pcd_path, result) in enumerate(zip(pcd_files, results_per_frame)):
        raw = read_pcd_xyzi_ascii(pcd_path)
        pts = preprocess_raw_for_second(raw)
        raw_points.append(pts)

        pred = result.pred_instances_3d
        box7, sc = filter_cyclist_from_mmdet(
            pred, score_thr=score_thr, model_name=model_name
        )
        boxes7_list[i] = box7
        scores_list[i] = sc

    # 2) Normalizza yaw sulla sequenza
    boxes7_norm = normalize_yaw_over_time(boxes7_list)

    # 3) Estrai punti usando box normalizzate
    pcd_list = [None] * T
    center_list = np.zeros((T, 3), dtype=np.float64)

    for i in range(T):
        box7 = boxes7_norm[i]
        if box7 is None:
            continue

        pts = raw_points[i]
        pts_ng = remove_ground_ransac(pts, distance_thresh=0.05)
        mask = points_in_oriented_box(pts_ng[:, :3], box7, pad=(0.25, 0.25, 0.5))
        inside = pts_ng[mask]

        pcd_list[i] = inside if len(inside) > 0 else None
        center_list[i] = box7[:3]

    return pcd_list, center_list, boxes7_norm, scores_list

def export_bike_pcd(pcd_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for i, pts in enumerate(pcd_list, start=1):
        if pts is None:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))

        fn = os.path.join(out_dir, f"pcd_bike_{i:03d}.pcd")
        o3d.io.write_point_cloud(fn, pcd, write_ascii=True)

def find_center(points_xyzi):
    return points_xyzi[:, :3].mean(axis=0)

def get_distance_and_speed(pcd_list, fps=10.0):
    """
    replica logica MATLAB: usa il centro dei punti (non il centro bbox).
    gestisce frame mancanti con count come nello script.
    """
    centers = []
    for pts in pcd_list:
        if pts is None:
            centers.append(None)
        else:
            centers.append(find_center(pts))

    dt = 1.0 / fps
    distances = np.zeros(max(len(centers)-1, 0), dtype=np.float64)
    speeds = np.zeros_like(distances)

    count = 1
    for i in range(len(centers)-1):
        c1 = centers[i]
        c2 = centers[i+1]
        if c1 is None:
            continue
        if c2 is None:
            count += 1
            continue

        distances[i] = np.linalg.norm(c1 - c2)
        speeds[i] = distances[i] / (dt * count) * 3.6  # m/s -> km/h (3.6)
        count = 1

    covered = float(np.sum(distances))
    speed_max = float(np.max(speeds)) if speeds.size else 0.0
    speed_mean = float(np.mean(speeds)) if speeds.size else 0.0
    return covered, speeds, speed_max, speed_mean, distances


def fit_plane_ransac(points_xyz, dist=0.1, ransac_n=3, iters=200):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    if len(pcd.points) < 30:
        return None  # insufficient
    plane, inliers = pcd.segment_plane(distance_threshold=dist,
                                       ransac_n=ransac_n,
                                       num_iterations=iters)
    # plane = [a,b,c,d] for ax+by+cz+d=0
    return np.array(plane, dtype=np.float64)

def angle_between_planes_deg(plane1, plane2):
    n1 = plane1[:3]; n2 = plane2[:3]
    den = (np.linalg.norm(n1)*np.linalg.norm(n2))
    if den < 1e-9:
        return np.nan
    cosang = np.clip(np.abs(np.dot(n1,n2))/den, 0.0, 1.0)
    angle = np.degrees(np.arccos(cosang))   # 0..90
    theta = 90.0 - angle                    #lean = 90 - angle_between_normals
    return theta



def split_bike_rider_kmeans(points_xyzi, random_state=1):
    P = points_xyzi[:, :3]
    z = P[:, 2].astype(np.float64)
    I = points_xyzi[:, 3].astype(np.float64) if points_xyzi.shape[1] > 3 else np.zeros_like(z)

    zf = (z - z.mean()) / (z.std() + 1e-9)
    If = (I - I.mean()) / (I.std() + 1e-9)
    X = np.c_[zf, If]

    km = KMeans(n_clusters=2, n_init=5, max_iter=500, random_state=random_state)
    idx = km.fit_predict(X)

    muZ = [z[idx==0].mean(), z[idx==1].mean()]
    bike_lbl = int(np.argmin(muZ))     # bici = cluster più basso in Z
    mask_bike = (idx == bike_lbl)
    bike = points_xyzi[mask_bike]
    rider = points_xyzi[~mask_bike]
    return bike, rider