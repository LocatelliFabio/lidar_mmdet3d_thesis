from pcd_reader import read_pcd_ascii
from show_points import visualize_points

pcd_path = "extracted_bike_pcd/pcd_bike_044.pcd"

raw_points = read_pcd_ascii(pcd_path)   # (N,4) float32

print("Punti raw:", len(raw_points))

visualize_points(raw_points)