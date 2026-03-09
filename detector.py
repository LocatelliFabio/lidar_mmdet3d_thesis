# detector.py

import numpy as np
import torch
from mmdet3d.apis import init_model, inference_detector


class SecondDetector:
    def __init__(self, config_path: str, checkpoint_path: str, device="cuda:0"):
        self.model = init_model(config_path, checkpoint_path, device=device)

    def infer(self, points: np.ndarray):
        if points is None or points.shape[0] == 0:
            return None

        with torch.no_grad():
            result, _ = inference_detector(self.model, points)
        return result

    @staticmethod
    def extract_predictions(result, score_thr=0.2):
        if result is None:
            return (
                np.empty((0, 7), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        pred = result.pred_instances_3d
        scores = pred.scores_3d.detach().cpu().numpy()
        labels = pred.labels_3d.detach().cpu().numpy()
        boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()

        keep = scores >= score_thr

        return (
            boxes[keep].astype(np.float32, copy=False),
            scores[keep].astype(np.float32, copy=False),
            labels[keep].astype(np.int64, copy=False),
        )