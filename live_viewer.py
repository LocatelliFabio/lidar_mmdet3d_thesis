import numpy as np
import open3d as o3d


LABEL_COLORS = {
    0: (0, 1, 0),
    1: (0, 0, 1),
    2: (1, 0, 0),
}


def create_oriented_boxes(boxes: np.ndarray, labels: np.ndarray):
    geometries = []

    for box, label in zip(boxes, labels):
        center = box[:3].copy()
        dims = box[3:6].copy()
        yaw = float(box[6])

        yaw = -yaw + np.pi / 2
        center[2] += dims[2] / 2.0

        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

        obb = o3d.geometry.OrientedBoundingBox(center, R, dims)
        obb.color = LABEL_COLORS.get(int(label), (1, 1, 1))
        geometries.append(obb)

    return geometries


class LiveViewer3D:
    def __init__(self, window_name="Live Detection Viewer", width=1280, height=720):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width, height)

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.box_geometries = []
        self.first_frame = True

        render = self.vis.get_render_option()
        render.point_size = 3.0
        render.background_color = np.array([0, 0, 0])

    def update(self, points: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        if points is not None and len(points) > 0:
            xyz = points[:, :3].astype(np.float64)
            self.pcd.points = o3d.utility.Vector3dVector(xyz)

            colors = np.ones((xyz.shape[0], 3), dtype=np.float64)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            self.vis.update_geometry(self.pcd)

        for g in self.box_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.box_geometries.clear()

        if boxes is not None and len(boxes) > 0:
            self.box_geometries = create_oriented_boxes(boxes, labels)
            for g in self.box_geometries:
                self.vis.add_geometry(g, reset_bounding_box=False)

        if self.first_frame and points is not None and len(points) > 0:
            self.vis.reset_view_point(True)
            self.first_frame = False

        self.vis.poll_events()
        self.vis.update_renderer()

    def spin_once(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()