from time import sleep
from threading import Lock
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

from rs_driver import (
    RSDriverParam,
    InputType,
    LidarType,
    LidarDriver,
    PointCloud,
    Error,
)

latest_points = None
latest_colors = None
points_lock = Lock()

cmap = cm.get_cmap("plasma")


def get_point_cloud_callback() -> PointCloud:
    return PointCloud()


def return_point_cloud_callback(point_cloud: PointCloud):
    global latest_points, latest_colors

    arr = point_cloud.numpy()  # (N, 4): x, y, z, intensity
    if arr is None or arr.shape[0] == 0:
        return

    xyz = arr[:, :3].astype(np.float64, copy=True)
    intensity = arr[:, 3].astype(np.float64, copy=True)

    xyz = xyz[::4]
    intensity = intensity[::4]

    # filtro opzionale: rimuove punti non finiti
    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
    xyz = xyz[mask]
    intensity = intensity[mask]

    if xyz.shape[0] == 0:
        return

    imin = float(intensity.min())
    imax = float(intensity.max())

    # evita divisione per zero se tutte le intensity sono uguali
    if imax > imin:
        intensity_norm = (intensity - imin) / (imax - imin)
    else:
        intensity_norm = np.zeros_like(intensity)

    colors = cmap(intensity_norm)[:, :3]

    with points_lock:
        latest_points = xyz
        latest_colors = colors

    print(
        f"points={xyz.shape[0]} "
        f"intensity_min={imin:.3f} intensity_max={imax:.3f}"
    )


def exception_callback(error: Error):
    print(f"Got error: {error}")


def main():
    global latest_points, latest_colors

    param = RSDriverParam()
    param.lidar_type = LidarType.RS128
    param.input_type = InputType.ONLINE_LIDAR
    param.input_param.msop_port = 6699
    param.input_param.difop_port = 7788
    param.input_param.host_address = "192.168.1.102"
    param.input_param.group_address = "0.0.0.0"

    driver = LidarDriver()
    driver.register_point_cloud_callback(
        get_point_cloud_callback,
        return_point_cloud_callback
    )
    driver.register_exception_callback(exception_callback)

    if not driver.init(param):
        print("Error initializing driver!")
        return

    driver.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RS128 Live Viewer", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    render = vis.get_render_option()
    render.point_size = 2.5
    render.background_color = np.array([1.0, 1.0, 1.0])  # bianco

    first_frame = True

    try:
        while True:
            pts = None
            cols = None

            with points_lock:
                if latest_points is not None and latest_colors is not None:
                    pts = latest_points.copy()
                    cols = latest_colors.copy()

            if pts is not None and pts.shape[0] > 0:
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)
                vis.update_geometry(pcd)

                if first_frame:
                    vis.reset_view_point(True)
                    first_frame = False

            vis.poll_events()
            vis.update_renderer()
            sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        vis.destroy_window()


if __name__ == "__main__":
    main()