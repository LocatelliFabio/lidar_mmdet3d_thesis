from time import sleep
from rs_driver import RSDriverParam, InputType, LidarType, LidarDriver, PointCloud, Error


def get_point_cloud_callback() -> PointCloud:
    return PointCloud()


def return_point_cloud_callback(point_cloud: PointCloud):
    print(f"Got: {point_cloud}")


def exception_callback(error: Error):
    print(f"Got error: {error}")


def main():
    param = RSDriverParam()
    param.lidar_type = LidarType.RSE1
    param.input_type = InputType.PCAP_FILE
    param.input_param.pcap_path = "path/to/file.pcap"
    param.print()

    driver = LidarDriver()
    driver.register_point_cloud_callback(get_point_cloud_callback, return_point_cloud_callback)
    driver.register_exception_callback(exception_callback)

    if not driver.init(param):
        print(f"Error initializing driver!")
        return

    driver.start()
    sleep(5)
    driver.stop()


if __name__ == "__main__":
    main()
