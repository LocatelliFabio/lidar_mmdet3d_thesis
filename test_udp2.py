import sys, os
import numpy as np
sys.path.append(os.path.abspath("bridge/build"))
from rs_bridge import RsBridge
from show_points import visualize_points
from preprocessing.pre_process import preprocess_raw_for_second
from mmdet3d.apis import init_model, inference_detector
from model_factory import get_model
from show_pcd import visualize_points_and_bboxes, visualize_bboxes_with_scores

bridge = RsBridge()
bridge.configure(host_ip="192.168.1.102", msop_port=6699, difop_port=7788, use_rsp128=True, split_angle_deg=0.0)
bridge.start()

points = None

for i in range(40):
    print("i = " + str(i))
    points = bridge.get_frame(timeout_ms=18000)


model = get_model()

if points is None:
    print("No frame")
    exit()

#visualize_points(points)

points = preprocess_raw_for_second(points)

result, data = inference_detector(model, points)

pred = result.pred_instances_3d
scores = pred.scores_3d.detach().cpu().numpy()
labels = pred.labels_3d.detach().cpu().numpy()
boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()

class_names = ['Pedestrian', 'Cyclist', 'Car']


#for cls_id, name in [(0,"Pedestrian"), (1,"Cyclist"), (2,"Car")]:
#    cls_scores = scores[labels == cls_id]
#    print(name, "count:", cls_scores.size,
#          "max:", (cls_scores.max() if cls_scores.size else None),
#          "top5:", np.sort(cls_scores)[-5:] if cls_scores.size else None)

    

visualize_bboxes_with_scores(points, result)