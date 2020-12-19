Generative Flows by PyTorch
===

> PyTorch implementations of the networks for generative flows or normalizing flows.

Models
---

Currently, following networks are implemented.

* RealNVP
  * Dinh et al., 2016, "Density Estimation using Real NVP," [[Link]](https://arxiv.org/abs/1605.08803)
* Glow
  * Kingma and Dhariwal 2018, "Glow: Generative Flow with Invertible 1x1 Convolutions," [[Link]](https://arxiv.org/abs/1807.03039v2)
* iResNet
  * Behrmann et al., 2018, "Invertible Residual Networks," [[Link]](https://arxiv.org/abs/1811.00995)
* Flow++
  * Ho et al., 2019, "Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design," [[Link]](https://arxiv.org/abs/1902.00275)
* FFJORD
  * Grathwohl et al., 2018, "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models," [[Link]](https://arxiv.org/abs/1810.01367)


Setup
---

### Anaconda

By Anaconda, you can easily setup the environment using `environment.yml`.

```shell
$ conda env create -f environment.yml
```

### Pip

If you use `pip` or other tools, see the dependencies in [`environment.yml`](./environment.yml)

Run
---

```shell
$ python main.py \
    network=[realnvp, glow, flow++, iresnet, ffjord]\
    train.distrib=[circles, moons, normals, swiss, s_curve, mnist, cifar10]
```

Screen shots
---

Comming soon...

Copyright
---

MIT License (c) 2020, Tatsuya Yatagawa
