# rs_driver_python

Python wrapper for the Robosense [rs_driver][] LiDAR kernel using [pybind11][].

The `rs_driver_python` package provides an easy way of accessing point cloud data from a Robosense LiDAR in Python by wrapping the C++ driver using pybind. This package bundles the `rs_driver`

**Note**: Currently only supports point clouds containing points of type PointXYZI (e.g. no rotating lidars). This is due to the limitation of having to bind generic C++ template arguments to separate Python classes.

### Dependencies

- Python (3.9+)
- pybind11 (2.13.6+)
- cmake (3.15+)

### Installation

Install the `rs_driver` dependencies:
```bash
sudo apt-get install libpcap-dev libeigen3-dev libboost-dev libpcl-dev
```

Download this repository and initialize the `rs_driver` submodule:
```bash
git clone https://gitlab.ub.uni-bielefeld.de/christopher.niemann/rs_driver_python.git
cd rs_driver_python
git submodule update --init --recursive
```
Build and install the python package with pip:
```bash
pip install .
```
The bundled `rs_driver` (from `external/rs_driver`) is automatically 
used to build the python bindings and does not have to be installed separately.

### Usage

The rs_driver python module corresponds directly to the C++ driver. Relevant API classes, structs and enums are directly mapped to Python and can be imported from the `rs_driver` module:
```python
from rs_driver import LidarDriver, RSDriverParam, LidarType, InputType, PointCloud
```

The driver parameters are set the exact same way as for the C++ driver:
```python
param = RSDriverParam()
param.lidar_type = LidarType.RSE1
param.input_type = InputType.PCAP_FILE
param.input_param.pcap_path = "/path/to/file.pcap"
```

Similarly, you can register the driver callbacks:
```python
driver = LidarDriver()
driver.register_point_cloud_callback(
    lambda: PointCloud(), 
    lambda point_cloud: print(f"Got: {point_cloud}"))
driver.register_exception_callback(lambda error: print(f"Got error {error}"))
```

Initialize and run the driver for 5 seconds:
```python
if driver.init(param)
    driver.start()
    sleep(5)
    driver.stop()
```

You can find a full example in `demo/demo_pcap.py`, a Python version of the [pcap demo] from the official rs_driver repository.


[rs_driver]: https://github.com/RoboSense-LiDAR/rs_driver
[pybind11]: https://pybind11.readthedocs.io
[pcap demo]: https://github.com/RoboSense-LiDAR/rs_driver/blob/ba07ba49565c8f1df745c8effaf64f39ac83a0ef/demo/demo_pcap.cpp
