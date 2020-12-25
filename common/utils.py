import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('Agg')

import numpy as np
import torch
import torchvision
import sklearn.datasets
import matplotlib.pyplot as plt
from PIL import Image


def scatter_plot(xs, ys, zs=None, colors=None, title=None):
    fig = plt.figure(figsize=(9, 8), tight_layout=True)
    canvas = FigureCanvas(fig)

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

    if title is not None:
        plt.title(title)

    plt.colorbar(scatter)

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype='uint8')[..., :3]
    plt.close()

    return image


def image_plot(image,
               cmap='inferno',
               vmin=0.0,
               vmax=1.0,
               extent=[-1, 1, -1, 1],
               axis=True,
               title=None,
               colorbar=True):

    fig = plt.figure(figsize=(9, 8), tight_layout=True)
    canvas = FigureCanvas(fig)

    ax = fig.add_subplot(111)
    im = ax.imshow(image, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(im)

    if not axis:
        plt.axis('off')

    if title is not None:
        plt.title(title)

    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype='uint8')[..., :3]
    plt.close()

    return image


def save_image(filename, image):
    if image.dtype == np.float32:
        image = (image * 255.0).astype('uint8')
    elif image.dtype != np.uint8:
        raise Exception('"save_image" only support uint8 or float32 types')

    image = Image.fromarray(image)
    image.save(filename)
