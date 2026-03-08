from time import sleep
from threading import Lock
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

latest_points = None
points_lock = Lock()


def get_point_cloud_callback() -> PointCloud:
    return PointCloud()


def return_point_cloud_callback(point_cloud: PointCloud):
    global latest_points

    arr = point_cloud.numpy()
    if arr is None or arr.shape[0] == 0:
        return

    # solo xyz
    xyz = arr[:, :3]

    # salva solo ultimo frame
    with points_lock:
        latest_points = xyz


def exception_callback(error: Error):
    # stampa solo errori importanti
    if "OVERFLOW" in str(error):
        print(error)


def main():
    global latest_points

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

    vis = o3d.visualization.Visualizer()
    vis.create_window("RS128 Live Viewer", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    render = vis.get_render_option()
    render.point_size = 2
    render.background_color = np.array([0, 0, 0])

    first_frame = True

    try:

        while True:

            with points_lock:
                if latest_points is None:
                    vis.poll_events()
                    vis.update_renderer()
                    continue

                pts = latest_points.copy()

            # sottocampionamento per velocità
            pts = pts[::4]

            pcd.points = o3d.utility.Vector3dVector(pts)

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