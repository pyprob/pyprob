# pyprob

`pyprob` is a [PyTorch](http://pytorch.org/)-based library for [probabilistic programming](http://probabilistic-programming.org) and inference compilation.

Inference compilation is a technique for performing fast inference in generative models implemented as probabilistic programs, using deep neural networks to parameterize proposal distributions of a sequential importance sampling inference engine.

## Installation

### Prerequisites:

* Python 3.5 or higher. We recommend [Anaconda](https://www.continuum.io/).
* Latest PyTorch, installed by following instructions on the [PyTorch web site](http://pytorch.org/).

### Install from source
To use a cutting-edge version, clone this repository and install the `pyprob` package using:

```
git clone git@github.com:probprog/pyprob.git
cd pyprob
pip install .
```

### Install using `pip`
To use the latest version available in [Python Package Index](https://pypi.python.org/), run:

```
pip install pyprob
```

## Usage

`pyprob` has two main modes of operation:

* Probabilistic programming and inference compilation fully in Python
* Interfacing with 3rd party probabilistic programming libraries (e.g., [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/index.html), CPProb) through a [ZeroMQ](http://zeromq.org/)/[FlatBuffers](https://google.github.io/flatbuffers/)-based protocol

### Probabilistic programming in Python

This is currently work in progress. A website with documentation and examples will be provided.

### Interfacing with 3rd party libraries
#### Compilation

After setting up the probabilistic program and initiating compilation mode in the 3rd party library, you can start the compilation module with the default set of parameters using:

```
pyprob-compile
```

This starts a training session with infinite training data supplied by the probabilistic model. You can stop training at any point by hitting Ctrl + C. Alternatively, you can use the `--maxTraces` option to stop after a set number of traces (e.g., `--maxTraces 1000`).

If you want to use GPU, you should install PyTorch with CUDA support and use the `--cuda` flag.

By default the compilation artifacts are saved to the current directory. This can be changed by using the `--dir` option (e.g., `--dir ~/artifacts`).

There are a number of parameters for configuring the compilation session, such as setting different embedding types and neural network architectures. For information on the various command line options available for compilation, use the `--help` flag.

#### Inference

After setting up the probabilistic program and initiating inference mode in the 3rd party library, you can start the inference module with the default set of parameters using:

```
pyprob-infer
```

This starts an inference session using the latest saved artifact in the current directory. The directory for loading the artifact from can be changed using the `--dir` option (e.g., `--dir ~/artifacts`). Inference can be run on GPU using the `--cuda` flag.

Use the `--help` flag to see all available options and functionality.

#### Analytics

You can use

```
pyprob-analytics
```

for showing various statistics about the latest saved artifact in the current directory. You can use the `--help` flag to see available options for other functionality including the production of loss plots and detailed analytics reports.

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

`pyprob` is distributed under the MIT License.

## Authors

`pyprob` has been developed by [Tuan Anh Le](http://www.tuananhle.co.uk/) and [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/) within the [Probabilistic Programming Group at the University of Oxford](https://github.com/probprog), led by [Frank Wood](http://www.robots.ox.ac.uk/~fwood/index.html).

For the full list of contributors, see:

https://github.com/probprog/pyprob/graphs/contributors
