# Torch library for Compiled Inference in the Probabilistic Programming System Anglican

## Paper
- [Inference Compilation and Probabilistic Programming](https://arxiv.org/abs/1610.09900)

## Installation
- [Clojure](http://clojure.org/guides/getting_started)
- [Leiningen](http://leiningen.org/#install)
- [Anglican CSIS](https://github.com/tuananhle7/anglican-csis)
- [Torch](http://torch.ch/)
- [Torch Autograd](https://github.com/twitter/torch-autograd)
- etc. (TODO)

## Usage

General info about the software setup...

We provide a minimal example in a [Gorilla worksheet](http://gorilla-repl.org/). To load it, run `lein gorilla` from the `examples` folder and load the worksheet `worksheet/minimal.clj`.

Check out a more in-depth [tutorial](TODO).

### Setting up Leiningen Project
Include the dependencies for [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/index.html) and the CSIS backend in your Leiningen `project.clj` file:
```
:dependencies [...
               [anglican "1.0.0"]
               [anglican-csis "0.1.0-SNAPSHOT"]
               ...])
```

In your Clojure file, remember to `require` the following in order to be able to define Anglican queries and perform CSIS:
```
  (:require ...
            anglican.csis.csis
            [anglican.csis.network :refer :all]
            [anglican.inference :refer [infer]]
            ...)
  (:use [anglican emit runtime])
```

### Compilation -- Training Neural Networks
After you've defined your probabilistic program in [Anglican language](http://www.robots.ox.ac.uk/~fwood/anglican/language/index.html), you can compile it.

The typical workflow consists of these steps:
1. Define a function to combine observes
2. Start a Clojure-Torch [ZeroMQ](http://zeromq.org/) connection from the Clojure side
3. Train the neural network in Torch
4. Stop the training of the neural network
5. Stop the Clojure-Torch ZeroMQ connection

### Inference -- Compiled Sequential Importance Sampling
After you've compiled your query by training up a neural network, you can perform inference using the Compiled Sequential Importance Sampling algorithm. You will hopefully need much fewer particles in comparison to Sequential Monte Carlo to perform inference.

The typical workflow consists of these steps:
1. Start a Torch-Clojure ZeroMQ connection from the Torch side
2. Run inference from Clojure
3. Stop the Torch-Clojure ZeroMQ connection
3. Evaluate inference in Clojure
