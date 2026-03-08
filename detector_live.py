import numpy as np
from mmdet3d.apis import init_model, inference_detector
from preprocessing.pre_process import preprocess_raw_for_second


CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car']

CLASS_COLORS = {
    0: np.array([0.0, 1.0, 0.0], dtype=np.float64),
    1: np.array([1.0, 0.0, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.4, 1.0], dtype=np.float64),
}


class LiveDetector:
    def __init__(self, config, checkpoint, device='cuda:0', score_thr=0.35):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.score_thr = score_thr
        self.model = None

    def load(self):
        self.model = init_model(self.config, self.checkpoint, device=self.device)

    def preprocess(self, raw_points: np.ndarray) -> np.ndarray:
        return preprocess_raw_for_second(raw_points)

    def infer(self, processed_points: np.ndarray):
        result, _ = inference_detector(self.model, processed_points)
        return result

    def extract_boxes(self, result):
        pred = result.pred_instances_3d

        scores = pred.scores_3d.detach().cpu().numpy()
        labels = pred.labels_3d.detach().cpu().numpy()
        boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()

        keep = scores >= self.score_thr

        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        detections = []
        for box, label, score in zip(boxes, labels, scores):
            detections.append({
                'label': int(label),
                'score': float(score),
                'box': box.astype(np.float64)
            })

        return detections