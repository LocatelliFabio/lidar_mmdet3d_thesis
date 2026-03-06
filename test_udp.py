import sys, os
sys.path.append(os.path.abspath("bridge/build"))
from rs_bridge import RsBridge
from show_pcd import visualize_points

bridge = RsBridge()
bridge.configure(host_ip="127.0.0.1", msop_port=6699, difop_port=7788, use_rsp128=True, split_angle_deg=0.0)
bridge.start()

points = bridge.get_frame(timeout_ms=20000)
points = bridge.get_frame(timeout_ms=20000)

if points is None:
    print("No frame")
    exit()

visualize_points(points)
#points = remove_invalid(points)
#points = crop_cpu(points)