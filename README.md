# PyTorch library for Inference Compilation and Universal Probabilistic Programming

This repository contains the [PyTorch](http://pytorch.org/)-based neural network component of `infcomp`, an inference compilation library. 

Inference compilation is a technique for performing fast inference in generative models implemented as [probabilistic programs](http://probabilistic-programming.org), using deep neural networks to parameterize proposal distributions of a sequential importance sampling inference engine.

The [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/)-based probabilistic programming component is available as a [Clojure library](https://github.com/probprog/anglican-inference-compilation). The PyTorch and Anglican components communicate over [ZeroMQ](http://zeromq.org/) using a [FlatBuffers](https://google.github.io/flatbuffers/)-based protocol.

For a walkthrough on how to set up a system to compile inference for a probabilistic program written in Anglican, check out the [tutorial](TUTORIAL.md). Also check out the [examples](https://github.com/probprog/torch-inference-compilation/tree/master/examples) folder.

## Installation

### Python `infcomp` package
Prerequisites:

* Python 3.5 or higher. We recommend [Anaconda](https://www.continuum.io/).
* Install latest PyTorch by following instructions on their [web site](http://pytorch.org/).

Clone this repository and install the Python `infcomp` package using:

```
git clone git@github.com:probprog/pytorch-infcomp.git
cd pytorch-infcomp
pip install .
```

## Usage

### Compilation

After setting up the probabilistic program and initiating compilation mode (see [tutorial](TUTORIAL.md)), you can start the compilation module with the default set of parameters using:

```
python -m infcomp.compile
```

This starts a training session with infinite training data supplied from the probabilistic model. You can stop training at any point by hitting Ctrl + C. Alternatively, you can use the `--maxTraces` option to stop after a set number of traces (e.g., `--maxTraces 1000`).

If you want to use GPU, you should install PyTorch with CUDA support and use the `--cuda` flag.

By default the compilation artifacts are saved to the current directory. This can be changed by using the `--dir` option (e.g., `--dir ~/artifacts`).

There are a number of parameters for configuring the compilation session, such as setting different embedding architectures and neural network dimensions. For information on the various command line options available on the PyTorch side, use the `--help` flag.

### Inference

After setting up the probabilistic program and initiating inference mode (see [tutorial(TUTORIAL.md)]), you can start the inference module with the default set of parameters using:

```
python -m infcomp.infer
```

This starts an inference session using the latest saved artifact in the current directory. The directory for loading the artifact from can be changed using the `--dir` option (e.g., `--dir ~/artifacts`). Inference can be run on GPU using the `--cuda` flag.

Use the `--help` flag to see all available options and functionality.

### Artifact information

You can use

```
python -m infcomp.info
```

for showing various statistics about the latest saved artifact in the current directory. You can use the `--help` flag to see available options for other functionality including the production of loss plots.

## Information on the technique, citing

Our [paper](https://arxiv.org/abs/1610.09900) provides an in-depth explanation of the inference compilation technique.

If you use this technique or the code, please use the following citation:
```
@inproceedings{le-2016-inference-compilation,
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

infcomp is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    infcomp is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with infcomp.  If not, see <http://www.gnu.org/licenses/>.    
