from pcd_reader import read_pcd_xyzi_ascii
from show_pcd import visualize_points_and_bboxes, visualize_bboxes_with_scores

if __name__ == "__main__":
    pcd_path = "extracted_bike_pcd/pcd_bike_011.pcd"
    points = read_pcd_xyzi_ascii(pcd_path)
    visualize_points_and_bboxes(points, result=None, score_thr=0.2)