def crop_pcd(points):
    pc_range=(0, -40, -3, 70.4, 40, 1)
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    m = (
        (x >= xmin) & (x <= xmax) &
        (y >= ymin) & (y <= ymax) &
        (z >= zmin) & (z <= zmax)
    )
    points = points[m]

    return points