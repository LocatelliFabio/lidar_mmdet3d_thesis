import numpy as np

def print_min_max(points):
    # min e max per colonna
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    print(f"x min: {mins[0]}, x max: {maxs[0]}")
    print(f"y min: {mins[1]}, y max: {maxs[1]}")
    print(f"z min: {mins[2]}, z max: {maxs[2]}")
    print(f"intensity min: {mins[3]}, intensity max: {maxs[3]}")

if __name__ == '__main__':
    # esempio: array N x 4
    points = np.array([
        [1.0, 2.0, 3.0, 10],
        [4.0, 5.0, -1.0, 20],
        [-2.0, 0.5, 6.0, 5]
    ])

    print_min_max(points)