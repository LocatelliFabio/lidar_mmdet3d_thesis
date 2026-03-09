import numpy as np
import sys, os
sys.path.append(os.path.abspath("bridge/build"))
from rs_bridge import RsBridge
from show_points import visualize_points

from print_min_max_nparray import print_min_max

bridge = RsBridge()
bridge.configure(host_ip="0.0.0.0", msop_port=6699, difop_port=7788, use_rsp128=False, split_angle_deg=0.0)
bridge.start()

print("Starting...")

points = bridge.get_frame(timeout_ms=7000)
points = bridge.get_frame(timeout_ms=7000)
#points = bridge.get_frame(timeout_ms=20000)

points = points[np.isfinite(points).all(axis=1)]

print_min_max(points)

if points is None:
    print("No frame")
    exit()

visualize_points(points)