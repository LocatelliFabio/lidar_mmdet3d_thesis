import numpy as np
from mmdet3d.apis import init_model, inference_detector
from pcd_reader import read_pcd_xyzi_ascii
from show_pcd import visualize_points_and_bboxes, visualize_bboxes_with_scores
from yaw_variants import preview_yaw_conventions
from preprocessing.pre_process import preprocess_raw_for_second
import time
import torch


def timed_pipeline(points, model, preprocess_fn, runs=10):
    """
    Misura tempi preprocess + inferenza.
    """

    # -----------------------
    # Warmup (GPU + CUDA)
    # -----------------------
    for _ in range(3):
        p = preprocess_fn(points)
        _ = inference_detector(model, p)

    torch.cuda.synchronize()

    # -----------------------
    # Benchmark
    # -----------------------
    t_pre = []
    t_inf = []

    for _ in range(runs):

        # preprocess
        t0 = time.perf_counter()
        p = preprocess_fn(points)
        t1 = time.perf_counter()

        # inference
        result, _ = inference_detector(model, p)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        t_pre.append(t1 - t0)
        t_inf.append(t2 - t1)

    # -----------------------
    # Stats
    # -----------------------
    print("Points after preprocess:", len(p))
    print()

    print("Preprocess:")
    print("  mean:", np.mean(t_pre))
    print("  std :", np.std(t_pre))
    print("  min :", np.min(t_pre))
    print("  max :", np.max(t_pre))

    print()

    print("Inference:")
    print("  mean:", np.mean(t_inf))
    print("  std :", np.std(t_inf))
    print("  min :", np.min(t_inf))
    print("  max :", np.max(t_inf))

    print()
    print("FPS:", 1.0 / (np.mean(t_pre) + np.mean(t_inf)))

