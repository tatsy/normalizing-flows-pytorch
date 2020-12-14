import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sklearn.datasets


def sample(n, name='moons'):
    if name == 'moons':
        samples, _ = sklearn.datasets.make_moons(n, noise=0.08)
        samples = (samples - 0.5) / 2.0

    else:
        raise Exception('Unsupported type: %s' % type)

    return samples


def save_plot(filename, xs, ys, colors):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    scatter = ax.scatter(xs, ys, c=colors, cmap='jet', vmin=0.0)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    plt.colorbar(scatter)
    plt.savefig(filename)
    plt.close()
