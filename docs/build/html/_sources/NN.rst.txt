Neural Network Library
======================
This library enables the composition of different NN architectures for inference compilation.

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:





Inference Network
-----------------

.. autoclass:: pyprob.nn.inference_network_feedforward.InferenceNetworkFeedForward
    :members:
    :undoc-members:
    :show-inheritance:

Embedding
---------

.. autoclass:: pyprob.nn.embedding_feedforward.EmbeddingFeedForward
    :members:
    :undoc-members:
    :show-inheritance:

Proposal Networks
-----------------

Normal
~~~~~~

.. autoclass:: pyprob.nn.proposal_normal_normal.ProposalNormalNormal
    :members:
    :undoc-members:
    :show-inheritance:

Normal Mixture
~~~~~~~~~~~~~~

.. autoclass:: pyprob.nn.proposal_normal_normal_mixture.ProposalNormalNormalMixture
    :members:
    :undoc-members:
    :show-inheritance:

Poisson
~~~~~~~

.. autoclass:: pyprob.nn.proposal_poisson_truncated_normal_mixture.ProposalPoissonTruncatedNormalMixture
    :members:
    :undoc-members:
    :show-inheritance:

Uniform Beta
~~~~~~~~~~~~

.. autoclass:: pyprob.nn.proposal_uniform_beta.ProposalUniformBeta
    :members:
    :undoc-members:
    :show-inheritance:

Uniform Beta Mixture
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyprob.nn.proposal_uniform_beta_mixture.ProposalUniformBetaMixture
    :members:
    :undoc-members:
    :show-inheritance:

Uniform Truncated Normal Mixture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autoclass:: pyprob.nn.proposal_uniform_truncated_normal_mixture.ProposalUniformTruncatedNormalMixture
    :members:
    :undoc-members:
    :show-inheritance:

