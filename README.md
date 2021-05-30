<div align="left">
  <a href="https://pyprob.readthedocs.io/en/latest/"> <img height="80px" src="docs/source/_static/pyprob-logo-large.png"></a>
</div>

-----------------------------------------

[![Build Status](https://github.com/pyprob/pyprob/workflows/build/badge.svg)](https://github.com/pyprob/pyprob/actions)
[![codecov](https://codecov.io/gh/pyprob/pyprob/branch/master/graph/badge.svg)](https://codecov.io/gh/pyprob/pyprob)
[![PyPI version](https://badge.fury.io/py/pyprob.svg)](https://badge.fury.io/py/pyprob)
[![Documentation Status](https://readthedocs.org/projects/pyprob/badge/?version=latest)](https://pyprob.readthedocs.io/en/latest/?badge=latest)

PyProb is a [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming) system for simulators and high-performance computing (HPC), based on [PyTorch](http://pytorch.org/). The main focus of PyProb is on coupling existing simulation code bases with probabilistic inference with minimal intervention.

PyProb is currently a research prototype, with more documentation and examples on the way. Watch this space!

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

## Installation, documentation, and examples

https://pyprob.readthedocs.io

## Information and citing

If you would like to learn more about or cite the techniques PyProb uses, please see the following papers:

* Baydin, Atılım Güneş, Lei Shao, Wahid Bhimji, Lukas Heinrich, Lawrence F. Meadows, Jialin Liu, Andreas Munk, Saeid Naderiparizi, Bradley Gram-Hansen, Gilles Louppe, Mingfei Ma, Xiaohui Zhao, Philip Torr, Victor Lee, Kyle Cranmer, Prabhat, and Frank Wood. 2019. “Etalumis: Bringing Probabilistic Programming to Scientific Simulators at Scale.” In Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC19), November 17–22, 2019. [arXiv:1907.03382](https://arxiv.org/abs/1907.03382)

* Baydin, Atılım Güneş, Lukas Heinrich, Wahid Bhimji, Lei Shao, Saeid Naderiparizi, Andreas Munk, Jialin Liu, Bradley Gram-Hansen, Gilles Louppe, Lawrence Meadows, Philip Torr, Victor Lee, Prabhat, Kyle Cranmer, and Frank Wood. 2019. “Efficient Probabilistic Inference in the Quest for Physics Beyond the Standard Model.” In Advances in Neural Information Processing Systems 33 (NeurIPS). [arXiv:1807.07706](https://arxiv.org/abs/1807.07706)

* Le, Tuan Anh, Atılım Güneş Baydin, and Frank Wood. 2017. “Inference Compilation and Universal Probabilistic Programming.” In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 54:1338–1348. Proceedings of Machine Learning Research. Fort Lauderdale, FL, USA: PMLR. [arXiv:1610.09900](https://arxiv.org/abs/1610.09900)

## License

PyProb is distributed under the BSD License.

## Authors

PyProb has been developed by the following core team. For the full list of contributors, see https://github.com/pyprob/pyprob/graphs/contributors

* [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/)
* [Tuan Anh Le](http://www.tuananhle.co.uk/)
* [Andreas Munk](https://ammunk.com/)
* [Saeid Naderiparizi](https://www.cs.ubc.ca/~saeidnp/)
* Francesco Pinto
* [Lei Shao](https://www.intel.com/content/www/us/en/artificial-intelligence/bios/lei-shao.html)
* [Jialin Liu](https://sites.google.com/site/jailinliu/)
* [Lukas Heinrich](http://www.lukasheinrich.com/)
* [Wahid Bhimji](http://www.nersc.gov/about/nersc-staff/data-analytics-services/wahid-bhimji/)
* [Kyle Cranmer](http://theoryandpractice.org/)
* [Frank Wood](http://www.cs.ubc.ca/~fwood/index.html)
