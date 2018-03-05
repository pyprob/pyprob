# pyprob [![Build Status](https://travis-ci.org/probprog/pyprob.svg?branch=master)](https://travis-ci.org/probprog/pyprob)

`pyprob` is a [PyTorch](http://pytorch.org/)-based library for [probabilistic programming](http://probabilistic-programming.org) and inference compilation.

Inference compilation is a technique for performing fast inference in generative models implemented as probabilistic programs, using deep neural networks to parameterize proposal distributions of a sequential importance sampling inference engine.

# Installation

## Prerequisites:

* Python 3.5 or higher. We recommend [Anaconda](https://www.continuum.io/).
* Latest PyTorch, installed by following instructions on the [PyTorch web site](http://pytorch.org/).

## Install from source
To use a cutting-edge version, clone this repository and install the `pyprob` package using:

```
git clone git@github.com:probprog/pyprob.git
cd pyprob
pip install .
```

## Install using `pip`
To use the latest version available in [Python Package Index](https://pypi.python.org/), run:

```
pip install pyprob
```

# Docker

A CUDA + PyTorch + pyprob image with the latest passing commit is automatically pushed to `probprog/pyprob:latest`

https://hub.docker.com/r/probprog/pyprob/

# Usage

`pyprob` has two main modes of operation:

* Probabilistic programming and inference compilation fully in Python
* Interfacing with 3rd party probabilistic programming libraries (e.g., [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/index.html), CPProb) through a [ZeroMQ](http://zeromq.org/)/[FlatBuffers](https://google.github.io/flatbuffers/)-based protocol

## Probabilistic programming in Python

**NOTE**: This is currently a work in progress, and the code in this public repository is under development. A website with documentation and examples will be provided in due course.


# Information and citing

[Our paper](https://arxiv.org/abs/1610.09900) at [AISTATS 2017](http://www.aistats.org/) provides an in-depth description of the inference compilation technique.

If you use `pyprob` and/or would like to cite our paper, please use the following information:
```
@inproceedings{le-2016-inference,
  author = {Le, Tuan Anh and Baydin, Atılım Güneş and Wood, Frank},
  booktitle = {Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  title = {Inference Compilation and Universal Probabilistic Programming},
  year = {2017},
  volume = {54},
  pages = {1338--1348},
  series = {Proceedings of Machine Learning Research},
  address = {Fort Lauderdale, FL, USA},
  publisher = {PMLR}
}
```

# License

`pyprob` is distributed under the MIT License.

# Authors

`pyprob` has been developed by [Tuan Anh Le](http://www.tuananhle.co.uk/) and [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/) within the [Probabilistic Programming Group at the University of Oxford](https://github.com/probprog), led by [Frank Wood](http://www.robots.ox.ac.uk/~fwood/index.html).

For the full list of contributors, see:

https://github.com/probprog/pyprob/graphs/contributors
