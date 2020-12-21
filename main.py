import os
import time
import shutil
import argparse
from itertools import cycle

import hydra
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf, DictConfig
from torch.distributions.multivariate_normal import MultivariateNormal

from flows import Glow, Ffjord, Flowxx, RealNVP, ResFlow
from common.utils import save_plot, save_image, save_image_plot
from flows.dataset import FlowDataset
from common.logging import Logging

networks = {
    'realnvp': RealNVP,
    'glow': Glow,
    'flow++': Flowxx,
    'resflow': ResFlow,
    'ffjord': Ffjord,
}

# -----------------------------------------------
# logging
# -----------------------------------------------
logger = Logging(__file__)


# -----------------------------------------------
# train/eval model
# -----------------------------------------------
class Model(object):
    def __init__(self, dims=(2, ), cfg=None):
        if torch.cuda.is_available():
            self.device = torch.device('cuda', cfg.run.gpu)
        else:
            self.device = torch.device('cpu')

        self.dims = dims
        self.dimension = np.prod(dims)

        mu = torch.zeros(self.dimension, dtype=torch.float32, device=self.device)
        covar = 0.25 * torch.eye(self.dimension, dtype=torch.float32, device=self.device)
        self.normal = MultivariateNormal(mu, covar)

        self.net = networks[cfg.network.name](dims=self.dims, cfg=cfg)
        self.net.to(self.device)

        if cfg.optimizer.name == 'rmsprop':
            self.optim = torch.optim.RMSprop(self.net.parameters(),
                                             lr=cfg.optimizer.lr,
                                             weight_decay=cfg.optimizer.weight_decay)
        elif cfg.optimizer.name == 'adam':
            self.optim = torch.optim.Adam(self.net.parameters(),
                                          lr=cfg.optimizer.lr,
                                          weight_decay=cfg.optimizer.weight_decay)
        else:
            raise Exception('optimizer "%s" is currently not supported' % (cfg.optimizer.name))

    def train_on_batch(self, y):
        y = y.to(self.device)
        self.net.train()

        z, log_det_jacobian = self.net(y)
        z = z.view(y.size(0), -1)
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

        with torch.no_grad():
            y, log_det_jacobians = self.net.backward(z.view(-1, *self.dims))
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


@hydra.main(config_path='configs', config_name='default')
def main(cfg):
    # show parameters
    print('***** parameters ****')
    print(OmegaConf.to_yaml(cfg))
    print('*********************')
    print('')

    workdir = hydra.utils.get_original_cwd()

    # CuDNN backends
    if cfg.run.debug:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # setup output directory
    out_dir = os.path.join(workdir, cfg.run.output, cfg.network.name, cfg.run.distrib)
    os.makedirs(out_dir, exist_ok=True)

    # dataset
    dset = FlowDataset(cfg.run.distrib)
    data_loader = torch.utils.data.DataLoader(dset, batch_size=cfg.train.samples, shuffle=True)

    # setup train/eval model
    model = Model(dims=dset.dims, cfg=cfg)

    # resume from checkpoint
    start_step = 0
    if cfg.run.ckpt_path is not None:
        start_step = model.load_ckpt(cfg.run.ckpt_path) + 1

    # training
    step = start_step
    for data in cycle(data_loader):
        # training
        start_time = time.perf_counter()
        y = data
        z, loss = model.train_on_batch(y)
        elapsed_time = time.perf_counter() - start_time

        step += 1

        if step % (cfg.run.display * 10) == 0:
            # logging
            logger.info('[%d/%d] loss=%.5f [%.3f s/it]' %
                        (step, cfg.train.steps, loss.item(), elapsed_time))

        if step % (cfg.run.display * 500) == 0:
            # testing
            y, py = model.sample_y(max(100, cfg.train.samples))
            y = y.detach().cpu().numpy()
            py = py.detach().cpu().numpy()

            if dset.dtype == '2d':
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

            if dset.dtype == '3d':
                # plot latent samples
                pz = model.pz(z)
                z = z.detach().cpu().numpy()
                pz = pz.detach().cpu().numpy()
                xs = z[:, 0]
                ys = z[:, 1]
                zs = z[:, 2]

                out_file = os.path.join(out_dir, 'z_sample_{:06d}.jpg'.format(step))
                save_plot(out_file, xs, ys, zs, colors=pz)
                latest_file = os.path.join(out_dir, 'z_sample_latest.jpg')
                shutil.copyfile(out_file, latest_file)

                # save plot
                xs = y[:, 0]
                ys = y[:, 1]
                zs = y[:, 2]

                out_file = os.path.join(out_dir, 'y_sample_{:06d}.jpg'.format(step))
                save_plot(out_file, xs, ys, zs, colors=py)
                latest_file = os.path.join(out_dir, 'y_sample_latest.jpg')
                shutil.copyfile(out_file, latest_file)

            if dset.dtype == 'image':
                images = torch.from_numpy(y[:100])
                images = torch.clamp(images, -1.0, 1.0) * 0.5 + 0.5
                grid_image = torchvision.utils.make_grid(images, nrow=10, pad_value=1)
                grid_image = grid_image.permute(1, 2, 0).numpy()
                out_file = os.path.join(out_dir, 'y_image_{:06d}.jpg'.format(step))
                save_image(out_file, grid_image)
                latest_file = os.path.join(out_dir, 'y_image_latest.jpg')
                shutil.copyfile(out_file, latest_file)

            # save ckpt
            ckpt_file = os.path.join(out_dir, 'latest.pth')
            model.save_ckpt(step, ckpt_file)


if __name__ == '__main__':
    main()
