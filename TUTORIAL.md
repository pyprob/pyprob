# Tutorial for Inference Compilation and Universal Probabilistic Programming

This is a tutorial on setting up a system to compile inference for a probabilistic program. Check out the [main repo](https://github.com/tuananhle7/torch-csis) and [more examples](examples/).

## Contents
- [Requirements](#requirements)
    - [Manual Installation](#manual-installation)
    - [Docker Image](#docker-image)
- [Breaking Captcha](#breaking-captcha)
    - [Writing the Probabilistic Program](#writing-the-probabilistic-program)
    - [Compilation](#compilation)
    - [Inference](#inference)

## Requirements
### Manual Installation
- [Clojure](http://clojure.org/guides/getting_started): Anglican runs on Clojure.
- [Leiningen](http://leiningen.org/#install): Package manager for Clojure.
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
$ luarocks install ansicolors
$ luarocks install cephes
$ luarocks install hash
$ luarocks install https://raw.github.com/jucor/torch-distributions/master/distributions-0-0.rockspec
$ luarocks install lzmq
$ luarocks install lua-messagepack
$ luarocks install rnn
```

### Docker Image

## Breaking Captcha
The code for Captcha breaking that we will go through in this tutorial along with other examples can be found in the [examples folder](examples/). As a prerequisite, you will have to familiarize yourself with [probabilistic programming in Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/usage/index.html).

### Writing the Probabilistic Program
#### Setting up a Leiningen Project
Create a Leiningen project using the [anglican-csis template](https://github.com/tuananhle7/anglican-csis-template) (replace `examples` by your own project name):
```
$ lein new anglican-csis examples
```
This will set up a Leiningen project with
- `project.clj` which contains the required dependencies and sets up the main entry point for this project
- `src/examples/core.clj` which contains the `main` method, including all the flags to start and stop the compilation and inference servers to interact with the Torch part.
- `src/queries/minimal.clj` with the namespace `queries.minimal` which contains a probabilistic program/query `minimal` along a function `combine-observes-fn` which will be needed for compiled inference.
- `src/worksheets/minimal.clj` which is a Jupyter-like, Clojure-based [Gorilla notebook](http://gorilla-repl.org/). We will not cover the Gorilla-based workflow in this tutorial although you are welcome to try it in the [examples folder](examples/).

In general, you will have to create a Clojure namespace in a `.clj` file (usually inside `src/queries/`) with a subset of the following:
- [*Probabilistic program/query*](#probabilistic-programquery)
- For the **compilation** part:
    - [*Function to combine observes*](#function-to-combine-observes)
    - [*Function to combine samples*](#function-to-combine-samples)
    - [*Query arguments*](#query-arguments-for-compilation)
- For the **inference** part:
    - [*Query arguments*](#query-arguments-for-inference)
    - [*Observe embedder input*](#observe-embedder-input)

For our purposes, let's create a file `src/queries/captcha_wikipedia.clj` with the namespace `queries.captcha-wikipedia`.

#### Probabilistic program/query
For Captcha purposes, we will have to set up some helper functions for rendering Captchas and calculating the approximate Bayesian computation likelihoods (which involves setting up a random projection matrix to reduce a 50x200 Captcha image to a 500-dimensional vector). These things are done in the following files:
- [src/helpers/OxCaptcha.java](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/OxCaptcha.java) - an [open-source standalone Java-based Captcha renderer](https://github.com/gbaydin/OxCaptcha)
- [resources/random-projection-matrices.h5](https://github.com/tuananhle7/torch-csis/blob/master/examples/resources/random-projection-matrices.h5) - precomputed random projection matrices in hdf5 format.
- [src/helpers/hdf5.java](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/hdf5.clj) - functions for parsing random projection matrices from the hdf5 file.
- [project.clj](https://github.com/tuananhle7/torch-csis/blob/master/examples/project.clj) - setting up dependencies for [hdf5](https://github.com/tuananhle7/torch-csis/blob/master/examples/project.clj#L8) and the [Java file](https://github.com/tuananhle7/torch-csis/blob/master/examples/project.clj#L9)
- [src/helpers/captcha.clj](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha.clj) - method for [reducing dimensions](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha.clj#L20)
- [src/helpers/captcha_wikipedia.clj](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha_wikipedia.clj) - setting up the [renderer](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha_wikipedia.clj#L12) and the [ABC likelihood](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha_wikipedia.clj#L37)

We then use these helper functions to build our generative model in [src/queries/captcha_wikipedia.clj](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L7). This is just a standard Anglican `query` with one exception: you must specify each `sample` by its address as the first and distribution as the second distribution, e.g. `(sample "numletters" (uniform-discrete 8 11))` on [line 10](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L10). This requirement will be removed in future updates.

#### Function to combine observes
Specify a function `combine-observes-fn` (you can name it anything) that combines observes from a sample in a form suitable for Torch. This will be used in the next step when starting the Clojure-Torch connection. In order to write this function, you'll need to look at how a typical `observes` object from your query looks like. This can be done by running `(sample-observes-from-prior q q-args)` where `q` is your query name and `q-args` are the arguments to the query.

#### Function to combine samples

#### Query arguments for compilation

#### Query arguments for inference

#### Observe embedder input

### Compilation [TODO]

### Inference [TODO]
