from time import sleep
from rs_driver import RSDriverParam, InputType, LidarType, LidarDriver, PointCloud, Error


def get_point_cloud_callback() -> PointCloud:
    return PointCloud()


def return_point_cloud_callback(point_cloud: PointCloud):
    print(point_cloud)
    arr = point_cloud.numpy()
    print("shape:", arr.shape)
    if arr.shape[0] > 0:
        print(arr[:5])


def exception_callback(error: Error):
    print(f"Got error: {error}")
    try:
        print("error_code:", error.error_code)
        print("error_type:", error.error_code_type)
    except Exception:
        pass


def main():
    param = RSDriverParam()
    param.lidar_type = LidarType.RS128
    param.input_type = InputType.ONLINE_LIDAR

    param.input_param.msop_port = 6699
    param.input_param.difop_port = 7788
    param.input_param.host_address = "192.168.1.102"
    param.input_param.group_address = "0.0.0.0"

    param.print()

    driver = LidarDriver()
    driver.register_point_cloud_callback(
        get_point_cloud_callback,
        return_point_cloud_callback
    )
    driver.register_exception_callback(exception_callback)

    if not driver.init(param):
        print("Error initializing driver!")
        return

    print("Driver initialized")
    driver.start()
    print("Driver started")

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        driver.stop()


if __name__ == "__main__":
    main()