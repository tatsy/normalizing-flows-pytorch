import matplotlib

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

    else:
        raise Exception('Unsupported type: %s' % type)

    return samples


def save_plot(filename, xs, ys, colors):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    scatter = ax.scatter(xs, ys, c=colors, cmap='jet', vmin=0.0, vmax=5.0)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.colorbar(scatter)
    plt.savefig(filename)
    plt.close()


def save_image_plot(filename, image, cmap='inferno'):
    im = plt.imshow(image, cmap='inferno', extent=[-1, 1, -1, 1], vmin=0.0, vmax=5.0)
    plt.colorbar(im)
    plt.savefig(filename)
    plt.close()
