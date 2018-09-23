Probabilistic Programming eXecution protocol
============================================

PPX |Build Status|
==================

.. raw:: html

   <p align="left">

.. raw:: html

   </p>

``PPX`` is a cross-platform `**P**\ robabilistic
**P**\ rogramming <http://www.probabilistic-programming.org>`__
e\ **X**\ ecution protocol and API based on
`flatbuffers <https://google.github.io/flatbuffers/>`__. It is intended
as an open interoperability protocol between models and inference
engines implemented in different probabilistic programming languages.

Probabilistic programming is about the execution probabilistic models
under the control of inference engines, and ``PPX`` allows the model and
the inference engine to be \* implemented in different programming
languages and \* executed in separate processes and on separate machines
across networks.

``PPX`` is inspired by `ONNX <https://onnx.ai/>`__, the Open Neural
Network Exchange project allowing interoperability between major deep
learning frameworks.

Supported languages
-------------------

We provide ``PPX`` compiled to all programming languages officially
supported by flatbuffers. These are:

-  C++
-  C#
-  Go
-  Java
-  JavaScript
-  PHP
-  Python
-  TypeScript

License
-------

``PPX`` is distributed under the BSD License.

Authors
-------

``PPX`` has been developed by

-  `Atılım Güneş Baydin <http://www.robots.ox.ac.uk/~gunes/>`__
-  `Tuan Anh Le <http://www.tuananhle.co.uk/>`__
-  `Lukas Heinrich <http://www.lukasheinrich.com/>`__
-  `Kyle Cranmer <http://theoryandpractice.org/>`__
-  `Frank Wood <http://www.cs.ubc.ca/~fwood/index.html>`__

.. |Build Status| image:: https://travis-ci.org/probprog/ppx.svg?branch=master
   :target: https://travis-ci.org/probprog/ppx


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:

   ppx_functions
