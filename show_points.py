import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pcd_reader import read_pcd_xyzi_ascii

def visualize_points(points: np.ndarray):
    """
    Visualizza solo la point cloud.

    points: (N,4) -> x,y,z,intensity
    """

    # ---------------------------
    # 1️⃣ Estrai coordinate e intensity
    # ---------------------------
    xyz = points[:, :3]
    intensity = points[:, 3]

    # ---------------------------
    # 2️⃣ Colori basati su intensity
    # ---------------------------
    intensity_norm = (intensity - intensity.min()) / (intensity.ptp() + 1e-6)
    cmap = plt.get_cmap("viridis")
    colors = cmap(intensity_norm)[:, :3]  # RGB

    # ---------------------------
    # 3️⃣ Crea point cloud Open3D
    # ---------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ---------------------------
    # 4️⃣ Visualizza
    # ---------------------------
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    pcd_path = "extracted_bike_pcd/pcd_bike_047.pcd"
    points = read_pcd_xyzi_ascii(pcd_path)
    visualize_points(points)