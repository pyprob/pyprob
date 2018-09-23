Distributions
=============

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:

PyTorch Distributions
---------------------

Where possible `PyProb` uses PyTorch distributions.
:class:`torch.distributions.distribution.Distribution`.
However, there are some differences between the two libraries. See
:class:`~pyprob.distributions.distribution.Distribution` for more details.

.. automodule:: pyprob.distributions

PyProb Distributions
--------------------

Base Class
~~~~~~~~~~

.. autoclass:: pyprob.distributions.distribution.Distribution
    :members:
    :undoc-members:
    :show-inheritance:

Beta
~~~~
.. autoclass:: pyprob.distributions.beta.Beta
    :members:
    :undoc-members:
    :show-inheritance:

Categorical
~~~~~~~~~~~
.. autoclass:: pyprob.distributions.categorical.Categorical
    :members:
    :undoc-members:
    :show-inheritance:


Empirical
~~~~~~~~~
.. autoclass:: pyprob.distributions.empirical.Empirical
    :members:
    :undoc-members:
    :show-inheritance:



Normal
~~~~~~
.. autoclass:: pyprob.distributions.normal.Normal
    :members:
    :undoc-members:
    :show-inheritance:

Mixture
~~~~~~~
.. autoclass:: pyprob.distributions.mixture.Mixture
    :members:
    :undoc-members:
    :show-inheritance:

Poisson
~~~~~~~
.. autoclass:: pyprob.distributions.poisson.Poisson
    :members:
    :undoc-members:
    :show-inheritance:

Truncated Normal
~~~~~~~~~~~~~~~~
.. autoclass:: pyprob.distributions.truncated_normal.TruncatedNormal
    :members:
    :undoc-members:
    :show-inheritance:


Uniform
~~~~~~~
.. autoclass:: pyprob.distributions.uniform.Uniform
    :members:
    :undoc-members:
    :show-inheritance: