import numpy as np
import open3d as o3d


def create_oriented_boxes(boxes: np.ndarray):
    line_sets = []

    for box in boxes:
        x, y, z, dx, dy, dz, yaw = box

        center = np.array([x, y, z], dtype=np.float64)
        extent = np.array([dx, dy, dz], dtype=np.float64)

        rot = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw),  np.cos(yaw), 0.0],
            [0.0,          0.0,         1.0]
        ], dtype=np.float64)

        obb = o3d.geometry.OrientedBoundingBox(center, rot, extent)
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)

        colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(ls.lines), 1))
        ls.colors = o3d.utility.Vector3dVector(colors)

        line_sets.append(ls)

    return line_sets


class LiveViewer3D:
    def __init__(self, window_name="Live Detection Viewer", width=1280, height=720):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width, height)

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.bbox_geometries = []
        self.first_frame = True

        render = self.vis.get_render_option()
        render.point_size = 3.0
        render.background_color = np.array([0, 0, 0])

    def update(self, points: np.ndarray, boxes: np.ndarray):
        if points is not None and len(points) > 0:
            xyz = points[:, :3].astype(np.float64)
            self.pcd.points = o3d.utility.Vector3dVector(xyz)

            colors = np.ones((xyz.shape[0], 3), dtype=np.float64)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            self.vis.update_geometry(self.pcd)

        for g in self.bbox_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.bbox_geometries.clear()

        if boxes is not None and len(boxes) > 0:
            self.bbox_geometries = create_oriented_boxes(boxes)
            for g in self.bbox_geometries:
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