# Torch library for Compiled Inference in the Probabilistic Programming System Anglican
## Paper
- [Inference Compilation and Universal Probabilistic Programming](https://arxiv.org/abs/1610.09900)

## Dependencies
- [Clojure](http://clojure.org/guides/getting_started)
- [Leiningen](http://leiningen.org/#install)
- [Anglican CSIS](https://github.com/tuananhle7/anglican-csis)
- [Torch](http://torch.ch/)
- [Torch Autograd](https://github.com/twitter/torch-autograd)
- etc. (TODO)

## Usage

This repository contains the Torch files required to perform the compilation stage in the compiled inference scheme. In our setup, probabilistic program definition and inference is done in the [Anglican Probabilistic Programming System](http://www.robots.ox.ac.uk/~fwood/anglican/) and the inference compilation is done in [Torch](http://torch.ch/). The communication between these two is facilitated by [ZeroMQ](http://zeromq.org/).

We provide a minimal example in a [Gorilla worksheet](http://gorilla-repl.org/). To load it, run `lein gorilla` from the `examples` folder and load the worksheet `worksheet/minimal.clj`.

See an explanation of a typical workflow below (can be followed in conjunction with the minimal example). Check out a more in-depth [tutorial](TODO). Experiments from the paper are [here](TODO).

### Setting up Leiningen Project
Include the dependencies for [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/index.html) and the Compiled Sequential Importance Sampling (CSIS) backend in your Leiningen `project.clj` file:
```
:dependencies [...
               [anglican "1.0.0"]
               [anglican-csis "0.1.0-SNAPSHOT"]
               ...])
```

In your Clojure file, remember to `require` the following in order to be able to define Anglican queries and perform inference using CSIS:
```
(:require ...
          anglican.csis.csis
          [anglican.csis.network :refer :all]
          [anglican.inference :refer [infer]]
          ...)
(:use [anglican emit runtime])
```

### Compilation: Training Neural Networks
After you've defined your probabilistic program in [Anglican language](http://www.robots.ox.ac.uk/~fwood/anglican/language/index.html), you can compile it.

The typical workflow consists of these steps:

#### Define a function to combine observes
TODO

#### Start a Clojure-Torch [ZeroMQ](http://zeromq.org/) connection from the Clojure side
TODO

#### Train the neural network in Torch
TODO

#### Stop the training of the neural network
TODO

#### Stop the Clojure-Torch ZeroMQ connection
TODO

### Inference: Compiled Sequential Importance Sampling
After you've compiled your query by training up a neural network, you can perform inference using the Compiled Sequential Importance Sampling algorithm. You will hopefully need much fewer particles in comparison to Sequential Monte Carlo to perform inference.

The typical workflow consists of these steps:

#### Start a Torch-Clojure ZeroMQ connection from the Torch side
TODO

#### Run inference from Clojure
TODO

#### Stop the Torch-Clojure ZeroMQ connection
TODO

#### Evaluate inference in Clojure
TODO

## Documentation
- [Documentation for the Clojure side](http://tuananhle.co.uk/anglican-csis-doc/)
- Documentation for the Torch side: run `compile.lua` and `infer.lua` with the `--help` flag.
