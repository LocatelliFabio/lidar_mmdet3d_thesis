import numpy as np
import open3d as o3d


CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']

CLASS_COLORS = {
    0: np.array([0.0, 1.0, 0.0], dtype=np.float64),  # Pedestrian
    1: np.array([1.0, 0.0, 0.0], dtype=np.float64),  # Cyclist
    2: np.array([0.0, 0.4, 1.0], dtype=np.float64),  # Car
}


def box7d_to_corners(box7d: np.ndarray) -> np.ndarray:
    """
    box7d = [x, y, z, dx, dy, dz, yaw]
    Assunzione: z è sul fondo del box -> centro geometrico = z + dz/2
    """
    x, y, z, dx, dy, dz, yaw = box7d

    cz = z + dz / 2.0

    corners = np.array([
        [ dx / 2,  dy / 2,  dz / 2],
        [ dx / 2, -dy / 2,  dz / 2],
        [-dx / 2, -dy / 2,  dz / 2],
        [-dx / 2,  dy / 2,  dz / 2],
        [ dx / 2,  dy / 2, -dz / 2],
        [ dx / 2, -dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, -dz / 2],
        [-dx / 2,  dy / 2, -dz / 2],
    ], dtype=np.float64)

    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    corners = (corners @ rot.T) + np.array([x, y, cz], dtype=np.float64)
    return corners


def build_lineset_from_detections(detections):
    all_points = []
    all_lines = []
    all_colors = []

    box_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    point_offset = 0

    for det in detections:
        label = det['label']
        box = det['box']
        color = CLASS_COLORS.get(label, np.array([1.0, 1.0, 1.0], dtype=np.float64))

        corners = box7d_to_corners(box)
        all_points.extend(corners.tolist())

        for line in box_lines:
            all_lines.append([line[0] + point_offset, line[1] + point_offset])
            all_colors.append(color.tolist())

        point_offset += 8

    ls = o3d.geometry.LineSet()

    if len(all_points) == 0:
        ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        return ls

    ls.points = o3d.utility.Vector3dVector(np.asarray(all_points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(all_lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(all_colors, dtype=np.float64))
    return ls


class LiveViewer:
    def __init__(self, width=1280, height=720, title="RS128 Live Detection", axis_size=2.0):
        self.width = width
        self.height = height
        self.title = title
        self.axis_size = axis_size

        self.vis = None
        self.pcd = None
        self.boxes = None
        self.axis = None
        self.first_frame = True

    def create(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(self.title, self.width, self.height)

        self.pcd = o3d.geometry.PointCloud()
        self.boxes = o3d.geometry.LineSet()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.axis_size,
            origin=[0.0, 0.0, 0.0]
        )

        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.boxes)
        self.vis.add_geometry(self.axis)

        render = self.vis.get_render_option()
        render.point_size = 2.0
        render.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, raw_points_xyz: np.ndarray, detections):
        """
        raw_points_xyz: (N,3)
        detections: lista di dict con label/score/box
        """
        if raw_points_xyz is not None and len(raw_points_xyz) > 0:
            self.pcd.points = o3d.utility.Vector3dVector(raw_points_xyz.astype(np.float64))
            self.vis.update_geometry(self.pcd)

        new_lineset = build_lineset_from_detections(detections)
        self.boxes.points = new_lineset.points
        self.boxes.lines = new_lineset.lines
        self.boxes.colors = new_lineset.colors
        self.vis.update_geometry(self.boxes)

        if self.first_frame and raw_points_xyz is not None and len(raw_points_xyz) > 0:
            self.vis.reset_view_point(True)
            self.first_frame = False

        self.vis.poll_events()
        self.vis.update_renderer()

    def destroy(self):
        if self.vis is not None:
            self.vis.destroy_window()