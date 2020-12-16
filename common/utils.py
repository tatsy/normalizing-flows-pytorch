import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


def sample(n, name='moons'):
    if name == 'moons':
        samples, _ = sklearn.datasets.make_moons(n, noise=0.08)
        samples = (samples - 0.5) / 2.0

    elif name == 'normals':
        radius = 0.7
        n_normals = 8
        k = np.random.randint(n_normals, size=(n))
        cx = radius * np.cos(2.0 * np.pi * k / n_normals)
        cy = radius * np.sin(2.0 * np.pi * k / n_normals)
        dx, dy = np.random.normal(size=(2, n)) * 0.1
        x = cx + dx
        y = cy + dy
        samples = np.stack([x, y], axis=1)

    elif name == 'swiss':
        samples, _ = sklearn.datasets.make_swiss_roll(n, noise=0.08)
        samples[:, 0] = samples[:, 0] * 0.07
        samples[:, 1] = samples[:, 1] * 0.07 - 1.0
        samples[:, 2] = samples[:, 2] * 0.07

    elif name == 's_curve':
        samples, _ = sklearn.datasets.make_s_curve(n, noise=0.08)
        samples[:, 0] = samples[:, 0] * 0.7
        samples[:, 1] = (samples[:, 1] - 1.0) * 0.7
        samples[:, 2] = samples[:, 2] * 0.35

    else:
        raise Exception('Unsupported type: "%s"' % name)

    return samples


def save_plot(filename, xs, ys, zs=None, colors=None):
    fig = plt.figure(figsize=(9, 8))
    if zs is None:
        # 2D
        ax = fig.add_subplot(111)
        scatter = ax.scatter(xs, ys, c=colors, cmap='jet')  #, vmin=0.0, vmax=5.0)
    else:
        # 3D
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=10.0, azim=-80.0)
        ax.set_zlim([-1, 1])
        ax.set_zticks(np.linspace(-1, 1, 5))
        scatter = ax.scatter(xs, ys, zs, c=colors, cmap='jet')  #, vmin=0.0, vmax=5.0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-1, 1])
    ax.set_xticks(np.linspace(-1, 1, 5))
    ax.set_ylim([-1, 1])
    ax.set_yticks(np.linspace(-1, 1, 5))
    plt.colorbar(scatter)
    plt.savefig(filename)
    plt.close()


def save_image_plot(filename, image, cmap='inferno'):
    im = plt.imshow(image, cmap=cmap, extent=[-1, 1, -1, 1])  #, vmin=0.0, vmax=5.0)
    plt.colorbar(im)
    plt.savefig(filename)
    plt.close()
