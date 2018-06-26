# pyprob [![Build Status](https://travis-ci.org/probprog/pyprob.svg?branch=master)](https://travis-ci.org/probprog/pyprob)

`pyprob` is a [PyTorch](http://pytorch.org/)-based library for [probabilistic programming](http://probabilistic-programming.org) and inference compilation. The main focus of this library is on coupling existing simulation codebases with probabilistic inference with minimal intervention.

`pyprob` is currently a research prototype in alpha testing stage, with more documentation and examples on the way. Watch this space!

### Support for multiple languages

We support front ends in multiple languages through a `pplprotocol` interface that allows execution of models and inference engines in separate programming languages, processes, and machines connected over a network. The currently supported languages are Python and C++.

* Python: `pyprob` is implemented and directly usable from Python
* C++: A lightweight C++ front end is available through the [pyprob_cpp](https://github.com/probprog/pyprob_cpp) library

### Inference engines

`pyprob` currently provides the following inference engines:
* Markov chain Monte Carlo
  * Lightweight Metropolis Hastings (LMH)
  * Random-walk Metropolis Hastings (RMH)
* Importance sampling
  * Regular sequential importance sampling (proposals from prior)
  * Inference compilation

Inference compilation is an amortized inference technique for performing fast repeated inference using deep neural networks to parameterize proposal distributions in the importance sampling family of inference engines. We are planning to add other inference engines, e.g., from the variational inference family.

## Installation

### Prerequisites:

* Python 3.5 or higher. We recommend [Anaconda](https://www.continuum.io/).
* PyTorch 0.4.0 or higher, installed by following instructions on the [PyTorch web site](http://pytorch.org/).

### Install from source
To use a cutting-edge version, clone this repository and install the `pyprob` package using:

```
git clone git@github.com:probprog/pyprob.git
cd pyprob
pip install .
```

### Install using `pip`
To use the latest version available in [Python Package Index](https://pypi.org/project/pyprob/), run:

```
pip install pyprob
```

## Docker

A CUDA + PyTorch + pyprob image with the latest passing commit is automatically pushed to `probprog/pyprob:latest`

https://hub.docker.com/r/probprog/pyprob/

## Usage, documentation, and examples

A website with documentation and examples will be available in due course.

The [examples](https://github.com/probprog/pyprob/tree/master/examples) folder in this repository provides some working models and inference workflows as Jupyter notebooks.

An set of continuous integration [tests](https://github.com/probprog/pyprob/tree/master/tests) are available in this repository, including those checking for correctness of inference over a range of reference models and inference engines.

## Information and citing

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

## License

`pyprob` is distributed under the BSD License.

## Authors

`pyprob` has been developed by [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/) and [Tuan Anh Le](http://www.tuananhle.co.uk/) within the [Probabilistic Programming Group at the University of Oxford](https://github.com/probprog), led by [Frank Wood](http://www.robots.ox.ac.uk/~fwood/index.html).

For the full list of contributors, see:

https://github.com/probprog/pyprob/graphs/contributors
