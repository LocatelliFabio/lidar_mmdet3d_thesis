# live_viewer.py

import numpy as np
import open3d as o3d


LABEL_COLORS = {
    0: (0, 1, 0),  # Pedestrian
    1: (0, 0, 1),  # Cyclist
    2: (1, 0, 0),  # Car
}


def _make_box(box: np.ndarray, label: int) -> o3d.geometry.OrientedBoundingBox:
    center = box[:3].copy()
    dims = box[3:6].copy()
    yaw = float(box[6])

    yaw = -yaw + np.pi / 2
    center[2] += dims[2] / 2.0

    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, yaw])

    obb = o3d.geometry.OrientedBoundingBox(center, R, dims)
    obb.color = LABEL_COLORS.get(int(label), (1, 1, 1))
    return obb


class LiveViewer3D:
    def __init__(
        self,
        window_name="Live Detection Viewer",
        width=1280,
        height=720,
        point_size=1.5,
        camera_position=(0.0, 0.0, 2.0),
        camera_lookat=(15.0, 0.0, 0.5),
    ):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width, height)

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=2.0, origin=[0.0, 0.0, 0.0]
        # )
        # self.vis.add_geometry(self.axis)

        self.box_geometries = []

        render = self.vis.get_render_option()
        render.point_size = point_size
        render.background_color = np.array([0.0, 0.0, 0.0])

        self.camera_position = np.asarray(camera_position, dtype=np.float64)
        self.camera_lookat = np.asarray(camera_lookat, dtype=np.float64)

        self.camera_initialized = False

    def _set_initial_camera(self):
        ctr = self.vis.get_view_control()

        self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()

        # Open3D usa front come vettore scena -> camera
        front = self.camera_position - self.camera_lookat
        norm = np.linalg.norm(front)
        if norm < 1e-9:
            front = np.array([-1.0, 0.0, 0.12], dtype=np.float64)
        else:
            front = front / norm

        ctr.set_lookat(self.camera_lookat.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up([0.0, 0.0, 1.0])

        # zoom fisso interno: non serve metterlo in config
        ctr.set_zoom(0.03)

        self.vis.poll_events()
        self.vis.update_renderer()

        self.camera_initialized = True

    @staticmethod
    def _normalize_intensity(points: np.ndarray) -> np.ndarray:
        """
        Restituisce intensity in [0,1].

        Gestisce sia:
        - intensity già normalizzata in [0,1]
        - intensity raw in [0,255]
        """
        n = points.shape[0]

        if points.shape[1] < 4:
            return np.ones((n,), dtype=np.float64)

        i = points[:, 3].astype(np.float64, copy=False)

        if i.size == 0:
            return np.ones((n,), dtype=np.float64)

        # Se i valori sembrano raw (tipicamente > 1), li normalizziamo
        if np.nanmax(i) > 1.0:
            i = np.clip(i, 0.0, 255.0) / 255.0
        else:
            i = np.clip(i, 0.0, 1.0)

        return i

    @staticmethod
    def _intensity_to_pseudocolor(points: np.ndarray) -> np.ndarray:
        """
        Pseudo color map leggera basata su intensity:
        basso -> blu
        medio -> ciano
        alto -> bianco
        """
        n = points.shape[0]
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)

        i = LiveViewer3D._normalize_intensity(points)

        r = np.clip(2.0 * i - 1.0, 0.0, 1.0)
        g = 0.2 + 0.8 * i
        b = 0.2 + 0.8 * i

        return np.stack((r, g, b), axis=1)

    def update(self, points: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        if points is not None and points.shape[0] > 0:
            xyz = points[:, :3].astype(np.float64, copy=False)
            self.pcd.points = o3d.utility.Vector3dVector(xyz)

            colors = self._intensity_to_pseudocolor(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            self.vis.update_geometry(self.pcd)

        for geom in self.box_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.box_geometries.clear()

        if boxes is not None and boxes.shape[0] > 0:
            for box, label in zip(boxes, labels):
                obb = _make_box(box, int(label))
                self.vis.add_geometry(obb, reset_bounding_box=False)
                self.box_geometries.append(obb)

        if (not self.camera_initialized) and points is not None and points.shape[0] > 0:
            self._set_initial_camera()

        self.vis.poll_events()
        self.vis.update_renderer()

    def spin_once(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()