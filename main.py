import os
import shutil
import argparse

import numpy as np
import scipy as sp
import torch
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal

from flows import Glow, Flowxx, RealNVP, InvResNet
from flows.utils import sample, save_plot, save_image_plot

networks = {
    'realnvp': RealNVP,
    'glow': Glow,
    'flow++': Flowxx,
    'iresnet': InvResNet,
}


class Model(object):
    def __init__(self, net='realnvp', n_dims=2, n_layers=4):
        if torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')

        mu = torch.zeros(n_dims, dtype=torch.float32, device=self.device)
        covar = torch.eye(n_dims, dtype=torch.float32, device=self.device)
        self.normal = MultivariateNormal(mu, covar)

        self.net = networks[net](n_dims=n_dims, n_layers=n_layers)
        self.net.to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1.0e-4)

    def train_on_batch(self, y):
        y = y.to(self.device)
        self.net.train()

        z, log_det_jacobian = self.net(y)
        loss = -1.0 * torch.mean(self.normal.log_prob(z) + log_det_jacobian)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return z, loss

    def sample(self, n):
        z = self.normal.sample([n])
        z = z.to(self.device)
        y, log_det_jacobians = self.net.backward(z)
        log_p = self.normal.log_prob(z) - log_det_jacobians
        return y, torch.exp(log_p)

    def log_prob(self, y):
        y = y.to(self.device)
        z, log_det_jacobians = self.net(y)
        return self.normal.log_prob(z) + log_det_jacobians

    def prob(self, y):
        return torch.exp(self.log_prob(y))


def main():
    torch.backends.cudnn.benchmark = True

    # command line arguments
    parser = argparse.ArgumentParser(description='Flow-based generative models')
    parser.add_argument('-n', '--network', type=str, required=True, choices=networks.keys(), help='name of network')
    parser.add_argument('-E', '--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('-B', '--batchsize', type=int, default=1024, help='minibatch size')
    parser.add_argument('-N', '--n_samples', type=int, default=1024, help='#samples to be drawn')
    parser.add_argument('--dist_name',
                        default='moons',
                        choices=['moons', 'normals'],
                        help='name of target distribution')
    parser.add_argument('--output', type=str, default='outputs', help='output directory')
    args = parser.parse_args()

    # setup output directory
    out_dir = os.path.join(args.output, args.network)
    os.makedirs(out_dir, exist_ok=True)

    # setup train/eval model
    model = Model(net=args.network)

    for epoch in range(args.epochs):
        # training
        pbar = tqdm(range(100))
        for i in pbar:
            y = sample(args.n_samples, name=args.dist_name)
            y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
            z, loss = model.train_on_batch(y)
            pbar.set_description('epoch #{:d}: loss={:.5f}'.format(epoch + 1, loss.item()))

        # testing
        y, prob = model.sample(args.n_samples)
        y = y.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()
        xs = y[:, 0]
        ys = y[:, 1]

        out_file = os.path.join(out_dir, 'y_sample_{:06d}.jpg'.format(epoch + 1))
        save_plot(out_file, xs, ys, prob)
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
        prob = model.prob(y)
        prob = prob.detach().cpu().numpy()
        prob_map = prob.reshape((map_size, map_size))

        out_file = os.path.join(out_dir, 'y_dist_{:06d}.jpg'.format(epoch + 1))
        save_image_plot(out_file, prob_map)
        latest_file = os.path.join(out_dir, 'y_dist_latest.jpg')
        shutil.copyfile(out_file, latest_file)


if __name__ == '__main__':
    main()
