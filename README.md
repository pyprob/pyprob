<div align="center">
  <a href=""> <img height="120px" src="docs/source/_static/img/pyprob-logo-large.png"></a>
</div>

-----------------------------------------

[![Build Status](https://travis-ci.org/pyprob/pyprob.svg?branch=master)](https://travis-ci.org/pyprob/pyprob)
[![PyPI version](https://badge.fury.io/py/pyprob.svg)](https://badge.fury.io/py/pyprob)

PyProb is a [probabilistic programming](http://probabilistic-programming.org) system for simulators and high-performance computing (HPC), based on [PyTorch](http://pytorch.org/). The main focus of PyProb is on coupling existing simulation code bases with probabilistic inference with minimal intervention.

PyProb is currently a research prototype, with more
documentation and examples on the way. Watch this space!

### Support for multiple languages

We support front ends in multiple languages through the
[PPX](https://github.com/pyprob/ppx) interface that allows execution of models
and inference engines in separate programming languages, processes, and machines
connected over a network.

### Inference engines

PyProb currently provides the following inference engines:
* Markov chain Monte Carlo
  * Lightweight Metropolis Hastings (LMH)
  * Random-walk Metropolis Hastings (RMH)
* Importance sampling
  * Regular sequential importance sampling (proposals from prior)
  * Inference compilation

Inference compilation is an amortized inference technique for performing fast
repeated inference using deep neural networks to parameterize proposal
distributions for importance sampling. We are planning to add other inference engines, e.g., variational inference.

## Installation

### Prerequisites:

* Python 3.5 or higher. We recommend [Anaconda](https://www.continuum.io/).
* PyTorch 1.0.0 or higher, installed by following instructions on the [PyTorch
  web site](http://pytorch.org/).

### Install from source
To use a cutting-edge version, clone this repository and install the PyProb package using:

```
git clone https://github.com/pyprob/pyprob.git
cd pyprob
pip install .
```

### Install the latest package
To use the latest version available in [Python Package
Index](https://pypi.org/project/pyprob/), run:

```
pip install pyprob
```

## Docker

You can build a [Docker](https://hub.docker.com/search/?type=edition&offering=community) image locally as follows:
```
git clone https://github.com/pyprob/pyprob.git
cd pyprob
docker build -t pyprob .
```

An image with the latest passing commit is automatically pushed to `pyprob/pyprob:latest` at https://hub.docker.com/r/pyprob/pyprob/. You can pull this as follows:
```
docker pull pyprob/pyprob
```

## Documentation and examples

Documentation coming soon.

## Information and citing

If you would like to learn more about or cite the techniques PyProb uses, please see the following papers:

* Baydin, Atılım Güneş, Lei Shao, Wahid Bhimji, Lukas Heinrich, Lawrence F. Meadows, Jialin Liu, Andreas Munk, Saeid Naderiparizi, Bradley Gram-Hansen, Gilles Louppe, Mingfei Ma, Xiaohui Zhao, Philip Torr, Victor Lee, Kyle Cranmer, Prabhat, and Frank Wood. 2019. “Etalumis: Bringing Probabilistic Programming to Scientific Simulators at Scale.” In Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC19), November 17–22, 2019. [arXiv:1907.03382](https://arxiv.org/abs/1907.03382)
```
@inproceedings{baydin-2019-etalumis,
  title = {Etalumis: Bringing Probabilistic Programming to Scientific Simulators at Scale},
  author = {Baydin, Atılım Güneş and Shao, Lei and Bhimji, Wahid and Heinrich, Lukas and Meadows, Lawrence F. and Liu, Jialin and Munk, Andreas and Naderiparizi, Saeid and Gram-Hansen, Bradley and Louppe, Gilles and Ma, Mingfei and Zhao, Xiaohui and Torr, Philip and Lee, Victor and Cranmer, Kyle and Prabhat and Wood, Frank},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC19), November 17--22, 2019},
  year = {2019}
}
```

* Baydin, Atılım Güneş, Lukas Heinrich, Wahid Bhimji, Lei Shao, Saeid Naderiparizi, Andreas Munk, Jialin Liu, Bradley Gram-Hansen, Gilles Louppe, Lawrence Meadows, Philip Torr, Victor Lee, Prabhat, Kyle Cranmer, and Frank Wood. 2019. “Efficient Probabilistic Inference in the Quest for Physics Beyond the Standard Model.” In Advances in Neural Information Processing Systems 33 (NeurIPS). [arXiv:1807.07706](https://arxiv.org/abs/1807.07706)
```
@inproceedings{baydin-2019-quest-for-physics,
  title = {Efficient Probabilistic Inference in the Quest for Physics Beyond the Standard Model},
  author = {Baydin, Atılım Güneş and Heinrich, Lukas and Bhimji, Wahid and Shao, Lei and Naderiparizi, Saeid and Munk, Andreas and Liu, Jialin and Gram-Hansen, Bradley and Louppe, Gilles and Meadows, Lawrence and Torr, Philip and Lee, Victor and Prabhat and Cranmer, Kyle and Wood, Frank},
  booktitle = {Advances in Neural Information Processing Systems 33 (NeurIPS)},
  year = {2019}
}
```
* Le, Tuan Anh, Atılım Güneş Baydin, and Frank Wood. 2017. “Inference Compilation and Universal Probabilistic Programming.” In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 54:1338–1348. Proceedings of Machine Learning Research. Fort Lauderdale, FL, USA: PMLR. [arXiv:1610.09900](https://arxiv.org/abs/1610.09900)
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

PyProb is distributed under the BSD License.

## Authors

PyProb has been developed by the team

* [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/)
* [Tuan Anh Le](http://www.tuananhle.co.uk/)
* [Andreas Munk](https://ammunk.com/)
* [Saeid Naderiparizi](https://www.cs.ubc.ca/~saeidnp/)
* [Lei Shao](https://www.intel.com/content/www/us/en/artificial-intelligence/bios/lei-shao.html)
* [Jialin Liu](https://sites.google.com/site/jailinliu/)
* [Lukas Heinrich](http://www.lukasheinrich.com/)
* [Wahid Bhimji](http://www.nersc.gov/about/nersc-staff/data-analytics-services/wahid-bhimji/)
* [Kyle Cranmer](http://theoryandpractice.org/)
* [Frank Wood](http://www.cs.ubc.ca/~fwood/index.html)

For the full list of contributors, see: https://github.com/pyprob/pyprob/graphs/contributors
