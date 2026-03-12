import numpy as np

def write_pcd_xyzi_ascii(path, points: np.ndarray):
    """
    points: numpy array (N,4) -> x,y,z,intensity
    """
    assert points.shape[1] == 4, "Expected (N,4) array"

    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write(f"WIDTH {points.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {points.shape[0]}\n")
        f.write("DATA ascii\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")