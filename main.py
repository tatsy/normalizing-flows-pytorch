import argparse

import numpy as np
import scipy as sp
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

from flows import FlowXX, RealNVP
from flows.utils import sample, save_plot


class Model(object):
    def __init__(self, n_dims=2, n_layers=4):
        if torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')

        mu = torch.zeros(n_dims, dtype=torch.float32, device=self.device)
        covar = torch.eye(n_dims, dtype=torch.float32, device=self.device)
        self.normal = MultivariateNormal(mu, covar)

        self.net = FlowXX(n_dims=n_dims, n_layers=n_layers)
        self.net.to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=1.0e-4)

    def train_on_batch(self, y):
        y = y.to(self.device)
        self.net.train()

        z, log_det_jacobian = self.net(y)
        loss = -1.0 * torch.mean(
            self.normal.log_prob(z) + torch.sum(log_det_jacobian, dim=1))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return z, loss

    def eval_on_batch(self, z):
        z = z.to(self.device)
        self.net.eval()

        y, log_det_jacobian = self.net.backward(z)
        jacobian = torch.exp(torch.sum(log_det_jacobian, dim=1))

        return y, jacobian


def main():
    torch.backends.cudnn.benchmark = True

    # command line arguments
    parser = argparse.ArgumentParser(
        description='Flow-based generative models')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='training epochs')
    parser.add_argument('--batchsize',
                        type=int,
                        default=1024,
                        help='minibatch size')
    parser.add_argument('-n',
                        '--n_samples',
                        type=int,
                        default=1024,
                        help='#samples to be drawn')
    parser.add_argument('--dist_name',
                        default='moons',
                        choices=['moons'],
                        help='name of target distribution')
    args = parser.parse_args()

    # setup train/eval model
    model = Model()

    normal = sp.stats.multivariate_normal(np.zeros(2), np.eye(2))

    for epoch in range(args.epochs):
        # training
        pbar = tqdm(range(100))
        for i in pbar:
            y = sample(args.n_samples, name=args.dist_name)
            y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
            z, loss = model.train_on_batch(y)
            pbar.set_description('epoch #{:d}: loss={:.5f}'.format(
                epoch + 1, loss.item()))

        # testing
        z = np.random.multivariate_normal(np.zeros(2),
                                          np.eye(2),
                                          size=(args.n_samples))
        z = torch.tensor(z, dtype=torch.float32)
        y, jacobian = model.eval_on_batch(z)

        y = y.detach().cpu().numpy()
        jacobian = jacobian.detach().cpu().numpy()
        pdf = normal.pdf(z) / jacobian
        xs = y[:, 0]
        ys = y[:, 1]
        save_plot('res.jpg', xs, ys, pdf)


if __name__ == '__main__':
    main()
