# Torch library for Compiled Inference in the Probabilistic Programming System Anglican
## Dependencies
- [Clojure](http://clojure.org/guides/getting_started): Anglican runs on Clojure.
- [Leiningen](http://leiningen.org/#install): Package manager for Clojure programs.
- [Anglican CSIS](https://github.com/tuananhle7/anglican-csis): Required for the CSIS extensions of Anglican (will be moved to Clojars soon)
```
$ git clone https://github.com/tuananhle7/anglican-csis.git
$ cd anglican-csis
$ lein install
```
- [Torch](http://torch.ch/docs/getting-started.html)
- [Torch Autograd](https://github.com/twitter/torch-autograd#install)
- Torch packages via [LuaRocks](https://luarocks.org/):
```
$ luarocks install nn
$ luarocks install nngraph
$ luarocks install lzmq # https://github.com/zeromq/lzmq
$ luarocks install lua-messagepack
$ luarocks install ansicolors
$ # etc (TODO)
```

## Usage

This repository contains the Torch files required to perform the compilation stage in the compiled inference scheme. In our setup, probabilistic program definition and inference is done in the [Anglican Probabilistic Programming System](http://www.robots.ox.ac.uk/~fwood/anglican/) and the inference compilation is done in [Torch](http://torch.ch/). The communication between these two is facilitated by [ZeroMQ](http://zeromq.org/).

### I just want to break some Captchas [TODO]

Download one of the artifacts and unzip it to the `data/` folder:
- [Wikipedia Captcha](TODO)
- [Facebook Captcha](TODO)

From the root folder of this repository, run
```
th infer.lua --cuda --latest
```

### I just want to cluster some points

### I want to compile my own models

We provide a minimal example in a [Gorilla worksheet](http://gorilla-repl.org/). To load it, run `lein gorilla` from the `examples` folder and load the worksheet `worksheet/minimal.clj`.

See an explanation of a typical workflow below (can be followed in conjunction with the minimal example). Check out a more in-depth [tutorial](TODO). Experiments from the paper are [here](TODO).

#### Setting up Leiningen Project
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

#### Compilation: Training Neural Networks
After you've defined your probabilistic program in [Anglican language](http://www.robots.ox.ac.uk/~fwood/anglican/language/index.html), you can compile it.

The typical workflow consists of these steps:

##### Define a function to combine observes
Specify a function `combine-observes-fn` (you can name it anything) that combines observes from a sample in a form suitable for Torch. This will be used in the next step when starting the Clojure-Torch connection.

In order to write this function, you'll need to look at how a typical `observes` object from your query looks like. This can be done by running `(sample-observes-from-prior q q-args)` where `q` is your query name and `q-args` are the arguments to the query.

##### Start a Clojure-Torch [ZeroMQ](http://zeromq.org/) connection from the Clojure side
Start a Clojure-Torch connection from Torch via running:
```
(def torch-connection (start-torch-connection q q-args combine-observes-fn))
```
Remember to bind this to a variable, in this case `torch-connection`, which will be used later to stop this connection. You can provide optional arguments such as TCP endpoint port.

##### Train the neural network in Torch
To train the neural net, `cd` to the root folder and run
```
th compile --help
```
to find out what options to run with.

##### Stop the training of the neural network
This can be done by `Ctrl+C` from the terminal. How long should I train? There aren't any theoretical minima. If all your random variables are discrete, the minimum should be around 0. Otherwise, just iterate between Compilation and Inference.

##### Stop the Clojure-Torch ZeroMQ connection
To stop the Clojure-Torch server from Clojuser, use the previously bound `torch-connection` as follows:
```
(stop-torch-connection torch-connection)
```

#### Inference: Compiled Sequential Importance Sampling
After you've compiled your query by training up a neural network, you can perform inference using the Compiled Sequential Importance Sampling algorithm. You will hopefully need much fewer particles in comparison to Sequential Monte Carlo to perform inference.

The typical workflow consists of these steps:

##### Run inference from Clojure
`cd` to the root folder and run
```
th infer --help
```
to find out what options to run with. Stop this process by `Ctrl+C` after you're done performing inference in Clojure.

##### Evaluate inference in Clojure
To get 10 particles from the CSIS inference algorithm, run
```
(def num-particles 10)
(def csis-states (take num-particles (infer :csis q q-args)))
(take 10 csis-states)
```

## Documentation
- [Documentation for the Clojure side](http://tuananhle.co.uk/anglican-csis-doc/)
- Documentation for the Torch side: run `compile.lua` and `infer.lua` with the `--help` flag.

## Paper
If you use this code in your work, please consider citing our [paper](https://arxiv.org/abs/1610.09900):
```
@article{le2016inference,
  title = {Inference Compilation and Universal Probabilistic Programming},
  author = {Le, Tuan Anh and Baydin, Atilim Gunes and Wood, Frank},
  journal = {arXiv preprint arXiv:1610.09900},
  year = {2016}
}
```
