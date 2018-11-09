# pyprob [![Build Status](https://travis-ci.org/probprog/pyprob.svg?branch=master)](https://travis-ci.org/probprog/pyprob)

`pyprob` is a [PyTorch](http://pytorch.org/)-based library for [probabilistic
programming](http://probabilistic-programming.org) and inference compilation.
The main focus of this library is on coupling existing simulation codebases with
probabilistic inference with minimal intervention.

`pyprob` is currently a research prototype in alpha testing stage, with more
documentation and examples on the way. Watch this space!

## Why pyprob?

The main advantage of `pyprob`, compared against other probabilistic programming
languages like Pyro, is a fully automatic amortized inference procedure based on
importance sampling. `pyprob` only requires a generative model to be specified.
Particularly, `pyprob` allows for efficient inference using inference
compilation which trains a recurrent neural network as a proposal network.

In Pyro such an inference network requires the user to explicitly define the
control flow of the network, which is due to Pyro running the inference network
and generative model sequentially. However, in `pyprob` the generative model and
inference network runs concurrently. Thus, the control flow of the model is
directly used to train the inference network. This alleviates the need for
manually defining its control flow.

Additionally, Pyro does not currently support distributed training of the
inference compilation network, whereas `pyprob` does.

### Support for multiple languages

We support front ends in multiple languages through the
[PPX](https://github.com/probprog/ppx) interface that allows execution of models
and inference engines in separate programming languages, processes, and machines
connected over a network. The currently supported languages are Python and C++.

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

Inference compilation is an amortized inference technique for performing fast
repeated inference using deep neural networks to parameterize proposal
distributions in the importance sampling family of inference engines. We are
planning to add other inference engines, e.g., from the variational inference
family.

## Installation

### Prerequisites:

* Python 3.5 or higher. We recommend [Anaconda](https://www.continuum.io/).
* PyTorch 0.4.0 or higher, installed by following instructions on the [PyTorch
  web site](http://pytorch.org/).

### Install from source
To use a cutting-edge version, clone this repository and install the `pyprob` package using:

```
git clone git@github.com:probprog/pyprob.git
cd pyprob
pip install .
```

### Install using `pip`
To use the latest version available in [Python Package
Index](https://pypi.org/project/pyprob/), run:

```
pip install pyprob
```

## Docker

A CUDA + PyTorch + pyprob image with the latest passing commit is automatically
pushed to `probprog/pyprob:latest`

https://hub.docker.com/r/probprog/pyprob/

## Usage, documentation, and examples

The simplest way to get started with pyprob, is to import the `pyprob` package
and `Model` class.

```python
import pyprob
from pyprob import Model
```

`pyprob` gives access to the `sample` and `observe` statements, which explicitly
denotes latent and observable variables of the program. `Model` is a superclass
containing methods for performing inference about the program.

Any distributions needed for the program is imported from
`pyprob.distributions`:

```python
from pyprob.distributions import Normal, Categorical # etc...
```

For a complete list of supported distributions see
[pyprob/distributions](https://github.com/probprog/pyprob/tree/master/pyprob/distributions).

### Example of a generative model

An illustrative example is the *Gaussian with unknown mean*, which can be
written as a probabilistic program using `pyprob` in the following way,

```python
import math
import pyprob
from pyprob import Model
from pyprob.distributions import Normal

class GaussianUnknownMean(Model):
    def __init__(self):
        super().__init__(name="Gaussian with unknown mean") # give the model a name
        self.prior_mean = 1
        self.prior_stdd = math.sqrt(5)
        self.likelhood_stdd = math.sqrt(2)

    def forward(self): # Needed to specifcy how the generative model is run forward
        # sample the (latent) mean variable to be inferred:
        mu = pyprob.sample(Normal(self.prior_mean, self.prior_stdd)) # NOTE: sample -> denotes latent variables

        # define the likelihood
        likelihood = Normal(mu, self.likelhood_stdd)

        # Lets add two observed variables
        # -> the 'name' argument is used later to assignment values:
        pyprob.observe(likelihood, name='obs0') # NOTE: observe -> denotes observable variables
        pyprob.observe(likelihood, name='obs1')

        # return the latent quantity of interest
        return mu

model = GaussianUnknownMean()
```

The task is to infer the unknown mean `mu` given/conditioned on the two observed variables `obs0` and `obs1`.

### Performing inference

In order to perform inference about the `model` (i.e. infer the posterior of
`mu`) from the previous example, call the `posterior_distribution` method and assign values to the `observe` variables:

```python
# sample from posterior (100 samples)
posterior = model.posterior_distribution(
                                         num_traces=100, # the number of samples estimating the posterior
                                         inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING, # specify which inference engine to use
                                         observe={'obs0': 8, 'obs1': 9} # assign values to the observed values
                                         )

# sample mean
posterior_mean = posterior.mean
# sample standard deviation
posterior_stdd = posterior.stddev
```

#### Inferring more than one latent variable

In the *Gassian with unknown mean* example, only a single latent variable `mu`
was returned. In case several latent variables are being returned the `.map`
method controls those returned latent variables. The return value is also an
empirical distribution object with the `.mean` and `.steddev` methods. The
`.map` methods takes **anonymous** functions as arguments, which are applied to
each sample:

```python
import pyprob

...

class SomeGenerativeModel(Model):

    ...

    def forward(...):

        ...

        return (var_0, var_1, var_2, ...) # return the desired number of latent variables to be inferred

model = SomeGenerativeModel()
posterior = model.posterior_distribution(...)

posterior_first = posterior.map(lambda v: v[0]) # extract var_0
var_0_mean = posterior_first.mean

# map can also be used to apply general functions and evaluate the statistical result under the posterior
posterior_first_sqrt = posterior.map(lambda v: v[0]**2) # extract var_0**2
var_0_sqrt_mean = posterior_first_sqrt.mean
```

### Visualization of and sampling from the posterior

Visualizing the result of performing inference is easily done using a histogram
(see [matplotlib's
*hist*](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) for
additional options)

```python
# assume "posterior_first" is available by running above commands
posterior_first.plot_histogram(show=True, bins=4)
```

Once a posterior is found sampling from the empirical distribution is done using
the `.sample` method. Sampling from the empirical distribution is done with
respect to the sample weights.

```python
samples_first = [posterior_first.sample() for _ in range(1000)] # 1000 samples
```

### Using other inference engines

The four aforementioned inference engines can be invoked by setting the
`inference_engine` argument to one of the following:

```python
pyprob.InferenceEngine.IMPORTANCE_SAMPLING # importance sampling using the prior as proposal distribution
pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK # importance sampling using inference compilation
pyprob.InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS # Lightweight MH
pyprob.InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS # Random-walk MH
```

### Using Inference Compilation

In case of using `IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK`, you should first
set the specification of the inference network you want to use and let it train
for a while. Training the inference network is inference compilation which
results in better proposal distributions for importance sampling engine. The
syntax for defining the network is the following:

```python
pyprob.Model.learn_inference_network(self,
                                     num_traces=None,
                                     inference_network=InferenceNetwork.FEEDFORWARD,
                                     prior_inflation=PriorInflation.DISABLED,
                                     trace_store_dir=None,
                                     observe_embeddings={},
                                     batch_size=64,
                                     valid_size=64, valid_interval=5000,
                                     learning_rate=0.0001,
                                     weight_decay=1e-5,
                                     auto_save_file_name_prefix=None,
                                     auto_save_interval_sec=600)
```



As an example, this the following code trains an inference network for the
example in this document.

```python
model.learn_inference_network(num_trace=10000,
                              observe_embeddings={'obs0' : {'dim' : 10},
                                                  'obs1': {'dim' : 10}})
```

`learn_inference_network` should be provided with the following arguments:
- `num_traces`: Specifies the number of traces (samples from the generative
  model) to be used for training the inference network.
- `observe_embeddings`: Specifies network structure for observe embedding
  networks. It should be a dictionary for every observed variable name (defined
  by `name` argument to `observe` or `sample` statements) to its embedding
  network specification. The embedding network specification is itself a
  dictionary with a subset of the following keys:
  - `dim`: Specifies dimension of the embedding. Default value is 256.
  - `embedding`: Specifies the network type. By default, it is a fully connected
    network. It currently supports `util.ObserveEmbedding.FEEDFORWARD`,
    `util.ObserveEmbedding.CNN2D5C` and `util.ObserveEmbedding.CNN3D4C`. Please
    refer to
    [pyprob/nn/emdebbing_*.py](https://github.com/probprog/pyprob/tree/master/pyprob/nn/)
    for a list of supported network types and their definition.
  - `depth`: Specifies depth of the network. Default value is 2.
  - `reshape`: Specifies shape of the network input. By default, embedding
    network input has the same shape as the value sampled from corresponding
    `observe` statement's distribution.

Once the network is trained, you can set `IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK` as `inference_engine` in `model.posterior_distribution` for performing inference.

```python
# sample from posterior using importance sampling and inference network (100 samples)
posterior = model.posterior_distribution(
                                         num_traces=100, # the number of samples estimating the posterior
                                         inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, # specify which inference engine to use
                                         observe={'obs0': 8, 'obs1': 9} # assign values to the observed values
                                         )
```

### More examples

The [examples](https://github.com/probprog/pyprob/tree/master/examples) folder
(to come) in this repository provides some working models and inference
workflows as Jupyter notebooks.

A set of continuous integration
[tests](https://github.com/probprog/pyprob/tree/master/tests) are available in
this repository, including those checking for correctness of inference over a
range of reference models and inference engines.

## Information and citing

[Our paper](https://arxiv.org/abs/1610.09900) at [AISTATS
2017](http://www.aistats.org/) provides an in-depth description of the inference
compilation technique.

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

`pyprob` has been developed by [Atılım Güneş
Baydin](http://www.robots.ox.ac.uk/~gunes/) and [Tuan Anh
Le](http://www.tuananhle.co.uk/) within the Programming Languages and AI group
led by [Frank Wood](http://www.cs.ubc.ca/~fwood/index.html) at the University of
Oxford and University of British Columbia.

For the full list of contributors, see:

https://github.com/probprog/pyprob/graphs/contributors
