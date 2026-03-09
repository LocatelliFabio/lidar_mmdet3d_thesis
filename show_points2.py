import numpy as np
import matplotlib.pyplot as plt

def show_points_grid(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter dei punti
    sc = ax.scatter(
        x, y, z,
        c=intensity,
        cmap='viridis',
        s=80,
        edgecolors='k'
    )

    # Limiti includendo anche l'origine
    xmin, xmax = min(x.min(), 0), max(x.max(), 0)
    ymin, ymax = min(y.min(), 0), max(y.max(), 0)
    zmin, zmax = min(z.min(), 0), max(z.max(), 0)

    # Piccolo margine visivo
    mx = (xmax - xmin) * 0.1 if xmax > xmin else 1
    my = (ymax - ymin) * 0.1 if ymax > ymin else 1
    mz = (zmax - zmin) * 0.1 if zmax > zmin else 1

    xmin, xmax = xmin - mx, xmax + mx
    ymin, ymax = ymin - my, ymax + my
    zmin, zmax = zmin - mz, zmax + mz

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    # Mantieni proporzioni corrette
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))

    # Disegna gli assi cartesiani reali passanti per l'origine
    ax.plot([xmin, xmax], [0, 0], [0, 0], color='r', linewidth=2, label='X')
    ax.plot([0, 0], [ymin, ymax], [0, 0], color='g', linewidth=2, label='Y')
    ax.plot([0, 0], [0, 0], [zmin, zmax], color='b', linewidth=2, label='Z')

    # Etichette
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualizzazione 3D punti con assi cartesiani')

    # Griglia
    ax.grid(True)

    # Colorbar intensity
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Intensity')

    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    points = np.array([
        [ 0.6580899 ,  1.2493421 ,  6.0064354 , 23.],
        [ 0.33336577,  1.2003815 ,  6.5938616 , 23.],
        [-0.04966142,  1.1589539 ,  7.1522202 , 32.],
        [-0.51146674,  1.129809  ,  7.716665  , 42.],
        [ 0.39402515,  1.3278904 ,  4.63196   , 23.],
    ], dtype=np.float32)

    show_points_grid(points)