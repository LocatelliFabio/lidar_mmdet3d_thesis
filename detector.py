import numpy as np
from mmdet3d.apis import init_model, inference_detector


class SecondDetector:
    def __init__(self, config_path: str, checkpoint_path: str, device='cuda:0'):
        self.model = init_model(config_path, checkpoint_path, device=device)

    def infer(self, points: np.ndarray):
        if points is None or len(points) == 0:
            return None

        result, data = inference_detector(self.model, points)
        return result

    @staticmethod
    def extract_predictions(result, score_thr=0.3):
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
        return boxes[keep], scores[keep], labels[keep]