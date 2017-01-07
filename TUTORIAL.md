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
The purpose of this function is to transform observations that are generated from the joint generative model of the latents and observations from a query-specific format to a clean N-D array consumable by the neural network's *observe embedder* ([f_obs in Figure 3](https://arxiv.org/pdf/1610.09900v1.pdf#page=5)).

How do these query-specific observations look like? You can output a sample from the prior by running the
[anglican.csis.network/sample-observes-from-prior](https://github.com/tuananhle7/anglican-csis/blob/master/src/anglican/csis/network.clj#L105) function from a REPL or a Gorilla notebook to see. More precisely, it is a vector of maps, each of which of the form
```
{:time-index <time-index>
 :observe-address <observe-address>
 :observe-instance <observe-instance>
 :value <value>}
```
which corresponds to a sample from an `observe` statement when running the program.

In our case, this vector will only have one element because our program has [only one observe statement](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L20). Hence [this is sufficient](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L28) as the function to combine observes:
```
(defn combine-observes-fn [observes]
  (:value (first observes)))
```
Remember the name of the function `combine-observes-fn` as we will use it later.

#### Function to combine samples
This function is optional and unnecessary in this Captcha example.

The purpose of this is to transform the query-specific list of samples before feeding it to the LSTM. Such things might include reordering the sample values. This has been done in the [Gaussian Mixture Model](https://github.com/tuananhle7/torch-csis/tree/master/examples#2-gaussian-mixture-model-with-fixed-number-of-clusters) [examples](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/gmm_variable_number_of_clusters.clj#L48) to sort the cluster-specific latents (means and covariances) according to the mean's norm.

How do these query-specific samples look like? You can output a sample from the prior by running the
[anglican.csis.network/sample-samples-from-prior](https://github.com/tuananhle7/anglican-csis/blob/master/src/anglican/csis/network.clj#L111) function from a REPL or a Gorilla notebook to see. More precisely, it is a vector of maps, each of which of the form
```
{:time-index <time-index>
 :sample-address <sample-address>
 :sample-instance <sample-instance>
 :prior-dist-str <prior-dist-str>
 :proposal-name <proposal-name>
 :proposal-extra-params <proposal-extra-params>
 :value <value>}
```
which corresponds to a sample from an `sample` statement when running the program.

#### Query arguments for compilation
Query arguments for compilation can either:
- specified in the query's namespace as a Clojure variable or
- supplied as a command line argument in [edn format](https://github.com/edn-format/edn) during compilation when running the `main` function.

In the Captcha case, the argument is a baseline Captcha but since this is ignored during compilation, `[nil]` is sufficient and can be easily supplied directly from the command line.

#### Query arguments for inference
Similarly to query arguments for compilation, query arguments for inference can also be either:
- specified in the query's namespace as a Clojure variable or
- supplied as a command line argument in [edn format](https://github.com/edn-format/edn) during inference when running the `main` function.

In the Captcha case, we want to create a pipeline "png -> inference -> inference results". For this purpose, we create a Python script to convert `png` to `edn` in [src/helpers/io/png2edn.py](https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/io/png2edn.py). We will see its usage [later](#inference).

#### Observe embedder input
This is optional and if unspecified, a value that will be passed to the *observe embedder* ([f_obs in Figure 3](https://arxiv.org/pdf/1610.09900v1.pdf#page=5)) is the first  of the [query arguments during inference](#query-arguments-for-inference).

In the case of Captcha, first query argument _is_ the observe embedder input so we leave this unspecified. However, you can supply the observe embedder input directly; again either:
- specified in the query's namespace as a Clojure variable or
- supplied as a command line argument in [edn format](https://github.com/edn-format/edn) during inference when running the `main` function.

### Compilation [TODO]

### Inference [TODO]
