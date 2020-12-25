import os
import time
import shutil
import argparse

import hydra
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal

from flows import Glow, Ffjord, Flowxx, RealNVP, ResFlow
from flows.misc import anomaly_hook
from common.utils import image_plot, save_image, scatter_plot
from flows.dataset import FlowDataLoader
from flows.modules import Logit, Identity
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

        self.name = cfg.network.name
        self.dims = dims
        self.dimension = np.prod(dims)

        mu = torch.zeros(self.dimension, dtype=torch.float32, device=self.device)
        covar = torch.eye(self.dimension, dtype=torch.float32, device=self.device)
        self.normal = MultivariateNormal(mu, covar)

        in_act_fn = Logit if len(dims) == 3 else Identity
        self.net = networks[self.name](dims=self.dims, in_act_fn=in_act_fn, cfg=cfg)
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

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def train_on_batch(self, y):
        y = y.to(self.device)

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

    def report(self, writer, y_data, step=0, save_files=False):
        # set to evaluation mode
        self.net.eval()

        # prepare
        y_data = y_data.to(self.device)
        n_samples = y_data.size(0)
        if y_data.dim() == 2 and y_data.size(1) == 2:
            dtype = '2d'
        elif y_data.dim() == 2 and y_data.size(1) == 3:
            dtype = '3d'
        else:
            dtype = 'image'

        title = '%s_%d_steps' % (self.name, step)

        # testing
        if dtype == '2d':
            # plot latent samples
            z, _ = self.net(y_data)
            pz = self.pz(z)
            z = z.detach().cpu().numpy()
            pz = pz.detach().cpu().numpy()
            xs = z[:, 0]
            ys = z[:, 1]

            z_image = scatter_plot(xs, ys, colors=pz, title=title)
            writer.add_image('2d/train/z', z_image, step, dataformats='HWC')

            if save_image:
                out_file = 'z_sample_{:06d}.jpg'
                save_image(out_file, z_image)
                latest_file = 'z_sample_latest.jpg'
                shutil.copyfile(out_file, latest_file)

            # save plot
            y, py = self.sample_y(max(100, n_samples))
            y = y.detach().cpu().numpy()
            py = py.detach().cpu().numpy()
            xs = y[:, 0]
            ys = y[:, 1]

            y_image = scatter_plot(xs, ys, colors=py, title=title)
            writer.add_image('2d/test/y', y_image, step, dataformats='HWC')

            if save_image:
                out_file = 'y_sample_{:06d}.jpg'.format(step)
                save_image(out_file, y_image)
                latest_file = 'y_sample_latest.jpg'
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

            py = self.py(y)
            py = py.detach().cpu().numpy()
            py_map = py.reshape((map_size, map_size))

            map_image = image_plot(py_map, title=title)
            writer.add_image('2d/test/map', map_image, step, dataformats='HWC')

            if save_image:
                out_file = 'y_dist_{:06d}.jpg'.format(step)
                save_image(out_file, map_image)
                latest_file = 'y_dist_latest.jpg'
                shutil.copyfile(out_file, latest_file)

        if dtype == '3d':
            # plot latent samples
            z, _ = self.net(y_data)
            pz = self.pz(z)
            z = z.detach().cpu().numpy()
            pz = pz.detach().cpu().numpy()
            xs = z[:, 0]
            ys = z[:, 1]
            zs = z[:, 2]

            z_image = scatter_plot(xs, ys, zs, colors=pz, title=title)
            writer.add_image('3d/train/z', z_image, step, dataformats='HWC')

            if save_image:
                out_file = 'z_sample_{:06d}.jpg'.format(step)
                save_image(out_file, z_image)
                latest_file = 'z_sample_latest.jpg'
                shutil.copyfile(out_file, latest_file)

            # save plot
            y, py = self.sample_y(max(100, n_samples))
            y = y.detach().cpu().numpy()
            py = py.detach().cpu().numpy()
            xs = y[:, 0]
            ys = y[:, 1]
            zs = y[:, 2]

            y_image = scatter_plot(xs, ys, zs, colors=py, title=title)
            writer.add_image('3d/test/y', y_image, step, dataformats='HWC')

            if save_image:
                out_file = 'y_sample_{:06d}.jpg'.format(step)
                save_image(out_file, y_image)
                latest_file = 'y_sample_latest.jpg'
                shutil.copyfile(out_file, latest_file)

        if dtype == 'image':
            images = torch.from_numpy(y[:100])
            images = torch.clamp(images, 0.0, 1.0)
            grid_image = torchvision.utils.make_grid(images, nrow=10, pad_value=1)
            grid_image = grid_image.permute(1, 2, 0).numpy()
            writer.add_image('image/test/grid', grid_image, step, dataformats='HWC')

            if save_image:
                out_file = 'y_image_{:06d}.jpg'.format(step)
                save_image(out_file, grid_image)
                latest_file = 'y_image_latest.jpg'
                shutil.copyfile(out_file, latest_file)


@hydra.main(config_path='configs', config_name='default')
def main(cfg):
    # show parameters
    print('***** parameters ****')
    print(OmegaConf.to_yaml(cfg))
    print('*********************')
    print('')

    # dataset
    dataset = FlowDataLoader(cfg.run.distrib,
                             batch_size=cfg.train.samples,
                             total_steps=cfg.train.steps,
                             shuffle=True)

    # setup train/eval model
    model = Model(dims=dataset.dims, cfg=cfg)

    # summary writer
    writer = SummaryWriter('./')

    # CuDNN backends
    if cfg.run.debug:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        for submodule in model.net.modules():
            submodule.register_forward_hook(anomaly_hook)

    else:
        torch.backends.cudnn.benchmark = True

    # resume from checkpoint
    start_step = 0
    if cfg.run.ckpt_path is not None:
        start_step = model.load_ckpt(cfg.run.ckpt_path) + 1

    # training
    step = start_step
    for data in dataset:
        # training
        model.train()
        start_time = time.perf_counter()
        y = data
        z, loss = model.train_on_batch(y)
        elapsed_time = time.perf_counter() - start_time

        if step % (cfg.run.display * 10) == 0:
            # logging
            logger.info('[%d/%d] loss=%.5f [%.3f s/it]' %
                        (step, cfg.train.steps, loss.item(), elapsed_time))

        if step % (cfg.run.display * 100) == 0:
            writer.add_scalar('{:s}/train/loss'.format(dataset.dtype), loss.item(), step)
            save_files = step % (cfg.run.display * 1000) == 0
            model.report(writer, y, step=step, save_files=save_files)
            writer.flush()

        if step % (cfg.run.display * 1000) == 0:
            # save ckpt
            ckpt_file = 'latest.pth'
            model.save_ckpt(step, ckpt_file)

        # update for the next step
        step += 1


if __name__ == '__main__':
    main()
