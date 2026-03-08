import numpy as np

from rs_driver import (
    RSDriverParam,
    InputType,
    LidarType,
    LidarDriver,
    PointCloud,
    Error,
)


class RSLidarStream:
    def __init__(self, frame_buffer, lidar_cfg: dict):
        self.frame_buffer = frame_buffer
        self.lidar_cfg = lidar_cfg
        self.driver = LidarDriver()

    def _get_point_cloud_callback(self) -> PointCloud:
        return PointCloud()

    def _return_point_cloud_callback(self, point_cloud: PointCloud):
        arr = point_cloud.numpy()
        if arr is None or arr.shape[0] == 0:
            return

        # usa XYZI se disponibile
        if arr.shape[1] >= 4:
            points = arr[:, :4].astype(np.float32, copy=False)
        else:
            # se non hai intensity, aggiungila a zero
            xyz = arr[:, :3].astype(np.float32, copy=False)
            intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
            points = np.hstack((xyz, intensity))

        self.frame_buffer.set(points)

    def _exception_callback(self, error: Error):
        if "OVERFLOW" in str(error):
            print(error)

    def init(self):
        param = RSDriverParam()
        param.lidar_type = LidarType.RS128
        param.input_type = InputType.ONLINE_LIDAR

        param.input_param.msop_port = self.lidar_cfg["msop_port"]
        param.input_param.difop_port = self.lidar_cfg["difop_port"]
        param.input_param.host_address = self.lidar_cfg["host_address"]
        param.input_param.group_address = self.lidar_cfg["group_address"]

        self.driver.register_point_cloud_callback(
            self._get_point_cloud_callback,
            self._return_point_cloud_callback
        )
        self.driver.register_exception_callback(self._exception_callback)

        return self.driver.init(param)

    def start(self):
        self.driver.start()

    def stop(self):
        self.driver.stop()