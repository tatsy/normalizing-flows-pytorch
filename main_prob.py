import os
import time
import shutil
import argparse

import numpy as np
import scipy as sp
import torch
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal

from flows import Glow, Flowxx, RealNVP, InvResNet
from common.utils import sample, save_plot, save_image_plot
from common.logging import Logging

networks = {
    'realnvp': RealNVP,
    'glow': Glow,
    'flow++': Flowxx,
    'iresnet': InvResNet,
}

# -----------------------------------------------
# FLAGs
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Flow-based generative models')
parser.add_argument('--network',
                    type=str,
                    required=True,
                    choices=networks.keys(),
                    help='name of network')
parser.add_argument('--layers', type=int, default=4, help='number of transformation layers')
parser.add_argument('--steps', type=int, default=10000, help='training steps')
parser.add_argument('--n_samples', type=int, default=1024, help='#samples to be drawn')
parser.add_argument('--distrib',
                    default='moons',
                    choices=['moons', 'normals', 'swiss', 's_curve'],
                    help='name of target distribution')
parser.add_argument('--ckpt_path',
                    type=str,
                    default=None,
                    help='checkpoint file to resume training')
parser.add_argument('--output', type=str, default='outputs', help='output directory')
parser.add_argument('--display', type=int, default=1, help='frequency for making report')
FLAGS = parser.parse_args()

# -----------------------------------------------
# logging
# -----------------------------------------------
logger = Logging(__file__)


# -----------------------------------------------
# train/eval model
# -----------------------------------------------
class Model(object):
    def __init__(self, net='realnvp', n_dims=2, n_layers=4):
        if torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')

        mu = torch.zeros(n_dims, dtype=torch.float32, device=self.device)
        covar = 0.25 * torch.eye(n_dims, dtype=torch.float32, device=self.device)
        self.normal = MultivariateNormal(mu, covar)

        self.net = networks[net](n_dims=n_dims, n_layers=n_layers)
        self.net.to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1.0e-4, weight_decay=1.0e-5)

    def train_on_batch(self, y):
        y = y.to(self.device)
        self.net.train()

        z, log_det_jacobian = self.net(y)

        loss = -1.0 * torch.mean(self.normal.log_prob(z) + log_det_jacobian)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return z, loss

    def save_ckpt(self, step, filename):
        ckpt = {
            'net': self.net.state_dict(),
            'optim': self.optim.state_dict(),
            'step': step,
        }
        torch.save(ckpt, filename)

    def load_ckpt(self, filename):
        ckpt = torch.load(filename)
        self.net.load_state_dict(ckpt['net'])
        self.optim.load_state_dict(ckpt['optim'])
        epoch = ckpt['step']
        return epoch

    def sample_y(self, n):
        z = self.sample_z(n)
        z = z.to(self.device)
        y, log_det_jacobians = self.net.backward(z)
        log_p = self.normal.log_prob(z) - log_det_jacobians
        return y, torch.exp(log_p)

    def sample_z(self, n):
        return self.normal.sample([n])

    def log_py(self, y):
        y = y.to(self.device)
        z, log_det_jacobians = self.net(y)
        return self.log_pz(z) + log_det_jacobians

    def log_pz(self, z):
        return self.normal.log_prob(z)

    def py(self, y):
        return torch.exp(self.log_py(y))

    def pz(self, z):
        return torch.exp(self.log_pz(z))


def main():
    # CuDNN backends
    torch.backends.cudnn.benchmark = True

    # setup output directory
    out_dir = os.path.join(FLAGS.output, FLAGS.network)
    os.makedirs(out_dir, exist_ok=True)

    # setup train/eval model
    n_dims = sample(1, name=FLAGS.distrib).shape[1]
    model = Model(net=FLAGS.network, n_dims=n_dims, n_layers=FLAGS.layers)

    # resume from checkpoint
    start_step = 0
    if FLAGS.ckpt_path is not None:
        start_step = model.load_ckpt(FLAGS.ckpt_path) + 1

    # training
    for step in range(start_step, FLAGS.steps):
        # training
        start_time = time.perf_counter()
        y = sample(FLAGS.n_samples, name=FLAGS.distrib)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
        z, loss = model.train_on_batch(y)
        elapsed_time = time.perf_counter() - start_time

        if step % (FLAGS.display * 10) == 0:
            # logging
            logger.info('[%d/%d] loss=%.5f [%.3f s/it]' %
                        (step, FLAGS.steps, loss.item(), elapsed_time))

        if step % (FLAGS.display * 100) == 0:
            # testing
            y, py = model.sample_y(FLAGS.n_samples)
            y = y.detach().cpu().numpy()
            py = py.detach().cpu().numpy()

            if n_dims == 2:
                # plot latent samples
                pz = model.pz(z)
                z = z.detach().cpu().numpy()
                pz = pz.detach().cpu().numpy()
                xs = z[:, 0]
                ys = z[:, 1]

                out_file = os.path.join(out_dir, 'z_sample_{:06d}.jpg'.format(step))
                save_plot(out_file, xs, ys, colors=pz)
                latest_file = os.path.join(out_dir, 'z_sample_latest.jpg')
                shutil.copyfile(out_file, latest_file)

                # save plot
                xs = y[:, 0]
                ys = y[:, 1]

                out_file = os.path.join(out_dir, 'y_sample_{:06d}.jpg'.format(step))
                save_plot(out_file, xs, ys, colors=py)
                latest_file = os.path.join(out_dir, 'y_sample_latest.jpg')
                shutil.copyfile(out_file, latest_file)

                # 2D visualization
                map_size = 256
                ix = (np.arange(map_size) + 0.5) / map_size * 2.0 - 1.0
                iy = (np.arange(map_size) + 0.5) / map_size * -2.0 + 1.0

                ix, iy = np.meshgrid(ix, iy)
                ix = ix.reshape((-1))
                iy = iy.reshape((-1))
                y = np.stack([ix, iy], axis=1)
                y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
                py = model.py(y)
                py = py.detach().cpu().numpy()
                py_map = py.reshape((map_size, map_size))

                out_file = os.path.join(out_dir, 'y_dist_{:06d}.jpg'.format(step))
                save_image_plot(out_file, py_map)
                latest_file = os.path.join(out_dir, 'y_dist_latest.jpg')
                shutil.copyfile(out_file, latest_file)

            if n_dims == 3:
                # save plot
                xs = y[:, 0]
                ys = y[:, 1]
                zs = y[:, 2]

                out_file = os.path.join(out_dir, 'y_sample_{:06d}.jpg'.format(step))
                save_plot(out_file, xs, ys, zs, colors=py)
                latest_file = os.path.join(out_dir, 'y_sample_latest.jpg')
                shutil.copyfile(out_file, latest_file)

            # save ckpt
            ckpt_file = os.path.join(out_dir, 'latest.pth')
            model.save_ckpt(step, ckpt_file)


if __name__ == '__main__':
    main()
