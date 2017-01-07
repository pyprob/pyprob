# Tutorial for Inference Compilation and Universal Probabilistic Programming

This is a tutorial on setting up a system to compile inference for a probabilistic program. Check out the [main repo][torch-csis-repo-link] and [more examples][examples-link].

## Contents
- [Requirements](#requirements)
    - [Manual Installation](#manual-installation)
    - [Docker Image](#docker-image)
- [Breaking Captcha](#breaking-captcha)
    - [Writing the Probabilistic Program](#writing-the-probabilistic-program)
    - [Compilation](#compilation)
    - [Inference](#inference)
- [Command Line Options](#command-line-options)
    - [Clojure](#clojure)
    - [Torch](#torch)

## Requirements
### Manual Installation
- [Clojure](http://clojure.org/guides/getting_started): Anglican runs on Clojure.
- [Leiningen](http://leiningen.org/#install): Package manager for Clojure.
- [Anglican CSIS][anglican-csis-repo-link]: Required for the CSIS extensions of Anglican (will be moved to Clojars soon)
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
In this tutorial, we will go through a complete example of breaking [Wikipedia's Captcha][actual-wikipedia-captcha-link]. The code for this in the [examples folder][examples-link] which contains other examples in addition to this one. Depending on your time, you can either try to re-create things described in this tutorial or just look at the relevant code in the examples folder.

As a prerequisite, you should familiarize yourself with [probabilistic programming in Anglican][anglican-link].

### Writing the Probabilistic Program
#### Setting up a Leiningen Project
Create a Leiningen project using the [anglican-csis template][anglican-csis-template-link] (replace `examples` by your own project name):
```
$ lein new anglican-csis examples
```
This will set up a Leiningen project with
- `project.clj` which contains the required dependencies and sets up the main entry point for this project
- `src/examples/core.clj` which contains the `main` method to start the compilation and inference ZeroMQ sockets to interact with the Torch part and the relevant command line options.
- `src/queries/minimal.clj` with the namespace `queries.minimal` which contains a probabilistic program/query `minimal` along a function `combine-observes-fn` which will be needed for compiled inference.
- `src/worksheets/minimal.clj` which is a Jupyter-like, Clojure-based [Gorilla notebook][gorilla-repl-link]. We will not cover the Gorilla-based workflow in this tutorial although you are welcome to try it in the [examples folder][examples-link].

In general, you will have to create a Clojure namespace in a `.clj` file (usually inside `src/queries/`) with a subset of the following:
- [*Probabilistic program/query*](#probabilistic-programquery)
- For the **compilation** part:
    - [*Function to combine observes*](#function-to-combine-observes)
    - [*Function to combine samples*](#function-to-combine-samples)
    - [*Query arguments*](#query-arguments-for-compilation)
- For the **inference** part:
    - [*Query arguments*](#query-arguments-for-inference)
    - [*Observe embedder input*](#observe-embedder-input)


Let's start by creating a file [`src/queries/captcha_wikipedia.clj`][captcha-wikipedia-clj-link] with the [namespace `queries.captcha-wikipedia`][captcha-wikipedia-clj-namespace-line-link].

#### Probabilistic program/query
For Captcha purposes, we will set up helper functions for rendering Captchas and calculating the approximate Bayesian computation (ABC) likelihoods (which involves setting up a random projection matrix to reduce a 50x200 Captcha image to a 500-dimensional vector). These things are done in the following files:
- [src/helpers/OxCaptcha.java][oxcaptcha-java-link] - an [open-source standalone Java-based Captcha renderer][oxcaptcha-github-link]
- [resources/random-projection-matrices.h5][random-projection-matrices-h5-link] - pre-computed random projection matrices in [hdf5][hdf5-link] format.
- [src/helpers/hdf5.clj][hdf5-clj-link] - functions for parsing random projection matrices from the hdf5 file.
- [project.clj][project-clj-link] - setting up dependencies for [hdf5][project-clj-hdf5-line-link] and the [Java file][project-clj-java-line-link]
- [src/helpers/captcha.clj][helpers-captcha-clj-link] - method for [reducing dimensions][helpers-captcha-clj-reducedim-line-link]
- [src/helpers/captcha_wikipedia.clj][helpers-captcha-wikipedia-clj-link] - setting up the [renderer][helpers-captcha-wikipedia-clj-render-line-link] and the [ABC likelihood][helpers-captcha-wikipedia-clj-abc-line-link]

We then use these helper functions to build our generative model in [src/queries/captcha_wikipedia.clj][captcha-wikipedia-clj-query-line-link]. This is just a standard Anglican `query` with one exception: you must specify each `sample` by its address as the first and distribution as the second distribution, e.g. [`(sample "numletters" (uniform-discrete 8 11))`][captcha-wikipedia-clj-sample-line-link]. This requirement will be removed in future updates.

The generative model itself is straightforward:
- sample number of letters
- sample font size
- sample [kerning][kerning-link]
- successively sample letters from a predefined alphabet
- render image
- constrain the generative model using an ABC likelihood

#### Function to combine observes
The purpose of this function is to transform observations that are generated from the joint generative model of the latents and observations from a query-specific format to a clean N-D array consumable by the neural network's *observe embedder* ([f_obs in Figure 3][figure3-paper-link]).

How do these query-specific observations look like? You can output a sample from the prior by running the
[`anglican.csis.network/sample-observes-from-prior`][sample-observes-from-prior-line-link] function from a REPL or a Gorilla notebook to see. More precisely, it is a vector of maps, each of which of the form
```
{:time-index <time-index>
 :observe-address <observe-address>
 :observe-instance <observe-instance>
 :value <value>}
```
which corresponds to a sample from an `observe` statement when running the program.

In our case, this vector will only have one element because our program has [only one observe statement][captcha-wikipedia-clj-observe-line-link]. Hence [this is sufficient][captcha-wikipedia-clj-combine-observes-fn-link] as the function to combine observes:
```
(defn combine-observes-fn [observes]
  (:value (first observes)))
```
Remember the name of the function `combine-observes-fn` as we will use it later.

#### Function to combine samples
This function is optional and unnecessary in this Captcha example.

The purpose of this is to transform the query-specific list of samples before feeding it to the LSTM. Such things might include reordering the sample values. This has been done in the [Gaussian Mixture Model][gmm-fixed-number-of-clusters-clj-link] [examples-link][gmm-variable-number-of-clusters-clj-link] to sort the cluster-specific latents (means and covariances) according to the mean's norm.

How do these query-specific samples look like? You can output a sample from the prior by running the
[`anglican.csis.network/sample-samples-from-prior`][sample-samples-from-prior-line-link] function from a REPL or a Gorilla notebook to see. More precisely, it is a vector of maps, each of which of the form
```
{:time-index <time-index>
 :sample-address <sample-address>
 :sample-instance <sample-instance>
 :prior-dist-str <prior-dist-str>
 :proposal-name <proposal-name>
 :proposal-extra-params <proposal-extra-params>
 :value <value>}
```
which corresponds to a sample from a `sample` statement when running the program.

#### Query arguments for compilation
Query arguments for compilation can be either:
- specified in the query's namespace as a Clojure variable or
- supplied as a command line argument in [edn format][edn-link] during compilation when running the `main` function.

In the Captcha case, the argument is a baseline Captcha but since this is ignored during compilation, `[nil]` is sufficient and can be easily supplied directly from the command line.

#### Query arguments for inference
Similarly, query arguments for inference can also be either:
- specified in the query's namespace as a Clojure variable or
- supplied as a command line argument in [edn format][edn-link] during inference when running the `main` function.

In the Captcha case, we want to create a pipeline "png->inference->inference results". For this purpose, we create a Python script to convert a png file to edn format in [src/helpers/io/png2edn.py][png2edn-py-link]. We will see its usage [later](#inference).

#### Observe embedder input
This is optional and if unspecified, a value that will be passed to the *observe embedder* ([f_obs in Figure 3][figure3-paper-link]) is the first of the [query arguments during inference](#query-arguments-for-inference).

In the case of Captcha, first query argument is the observe embedder input so we leave this unspecified. However, you can supply the observe embedder input directly; again either:
- specified in the query's namespace as a Clojure variable or
- supplied as a command line argument in [edn format][edn-link] during inference when running the `main` function.

### Compilation
To start a reply server in the [ZeroMQ request-reply socket pair][zmq-rep-req-link], we can run the project from the command line using the `lein run`:
```
lein run -- \
--mode compile \
--namespace queries.captcha-wikipedia \
--query captcha-wikipedia \
--compile-combine-observes-fn combine-observes-fn \
--compile-query-args-value "[nil]"
```
This reply server supplies the Torch side with samples from the prior. See `lein run -- -h` for help.

At the same time, run another process from [torch-csis][torch-csis-repo-link] root:
```
th compile.lua --batchSize 8 --validSize 8 --validInterval 32 --obsEmb lenet --obsEmbDim 4 --lstmDim 4
```
This starts the neural network training, getting data through the request client in the ZeroMQ request-reply socket pair. Torch will keep the artifact (neural network architecture, parameters, etc.) with the best validation loss so far in a file specified through the `--artifact` option. By default, it will be `./data/compile-artifact-<datetime>`.

When satisfied, you cancel both processes in your command line to stop the compilation. Note that compilation can be resumed by running `th compile.lua --resume <artifact-name>` (or `th compile.lua --resumeLatest`) with appropriate additional options. See `th compile.lua --help` for help.

### Inference
To perform inference on a fresh Captcha png image, say `resources/wikipedia-dataset/agavelooms.png`, run:
```
lein run -- \
--mode infer \
--namespace queries.captcha-wikipedia \
--query captcha-wikipedia \
--infer-query-args-value "[$(python src/helpers/io/png2edn.py resources/wikipedia-dataset/agavelooms.png)]"
```
This runs sequential importance sampling, requesting proposal parameters whenever required through the ZeroMQ request client.

At the same time, start the corresponding ZeroMQ reply server by running the following from [torch-csis][torch-csis-repo-link] root:
```
th infer.lua --latest
```
where `--latest` refers to the latest trained artifact in `./data/` folder. To pick a specific artifact, use the `--artifact` option.

## Command Line Options
### Clojure
Descriptions of command line options can be viewed using the `--help` option from the Leiningen project folder:
```
$ lein run -- --help
```
Here we provide the same, with links to different parts in the tutorial:

Long opt | Short opt | Description
--- | --- | ---
`--help` | `-h` | Shows help
`--mode` | `-m` | Choose between [compilation](#compilation) (`compile`) or [inference](#inference) (`infer`) mode
`--namespace` | `-n` | Clojure namespace containing the probabilistic program and things needed for compiled inference
`--query` | `-q` | Name of the [probabilistic program/query](#probabilistic-programquery) for compiled inference
`--compile-tcp-endpoint` | `-t` | TCP address for the ZMQ [compilation](#compilation) reply server
`--compile-combine-observes-fn` | `-o` | [Function to combine observes](#function-to-combine-observes)
`--compile-combine-samples-fn` | `-s` | [Function to combine samples](#function-to-combine-samples)
`--compile-query-args` | `-a` | [Query arguments for compilation as Clojure variable](#query-arguments-for-compilation)
`--compile-query-args-value` | `-x` | [Query arguments for compilation in edn format](#query-arguments-for-compilation)
`--infer-number-of-samples` | `-N` | [Number of samples from sequential importance sampling](#inference)
`--infer-tcp-endpoint` | `-T` | TCP address for the ZMQ [inference](#inference) request client
`--infer-observe-embedder-input` | `-E` | [Observe embedder input as Clojure variable](#observe-embedder-input)
`--infer-observe-embedder-input-value` | `-Y` | [Observe embedder input in edn format](#observe-embedder-input)
`--infer-query-args` | `-A` | [Query arguments for inference as Clojure variable](#query-arguments-for-inference)
`--infer-query-args-value` | `-Z` | [Query arguments for inference in edn format](#query-arguments-for-inference)

### Torch
Descriptions of command line options can be viewed using the `--help` option from the [torch-csis][torch-csis-repo-link] root:
```
$ th compile.lua --help
$ th infer.lua --help
$ th artifact-info.lua --help
```

[examples-link]: examples/
[actual-wikipedia-captcha-link]: https://en.wikipedia.org/w/index.php?title=Special:CreateAccount&returnto=Main+Page
[anglican-link]: http://www.robots.ox.ac.uk/~fwood/anglican/usage/index.html
[captcha-wikipedia-clj-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj
[captcha-wikipedia-clj-namespace-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L1
[anglican-csis-template-link]: https://github.com/tuananhle7/anglican-csis-template
[gorilla-repl-link]: http://gorilla-repl.org/
[oxcaptcha-java-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/OxCaptcha.java
[oxcaptcha-github-link]: https://github.com/gbaydin/OxCaptcha
[random-projection-matrices-h5-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/resources/random-projection-matrices.h5
[hdf5-clj-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/hdf5.clj
[project-clj-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/project.clj
[project-clj-hdf5-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/project.clj#L8
[project-clj-java-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/project.clj#L9
[helpers-captcha-clj-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha.clj
[helpers-captcha-clj-reducedim-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha.clj#L20
[helpers-captcha-wikipedia-clj-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha_wikipedia.clj
[helpers-captcha-wikipedia-clj-render-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha_wikipedia.clj#L12
[hdf5-link]: TODO
[helpers-captcha-wikipedia-clj-abc-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/captcha_wikipedia.clj#L37
[captcha-wikipedia-clj-query-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L8
[captcha-wikipedia-clj-sample-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L10
[kerning-link]: https://en.wikipedia.org/wiki/Kerning
[figure3-paper-link]: https://arxiv.org/pdf/1610.09900v1.pdf#page=5
[sample-observes-from-prior-line-link]: https://github.com/tuananhle7/anglican-csis/blob/master/src/anglican/csis/network.clj#L105
[captcha-wikipedia-clj-observe-line-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L20
[captcha-wikipedia-clj-combine-observes-fn-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/captcha_wikipedia.clj#L28
[gmm-fixed-number-of-clusters-clj-link]: https://github.com/tuananhle7/torch-csis/tree/master/examples#2-gaussian-mixture-model-with-fixed-number-of-clusters
[gmm-variable-number-of-clusters-clj-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/queries/gmm_variable_number_of_clusters.clj#L48
[sample-samples-from-prior-line-link]: https://github.com/tuananhle7/anglican-csis/blob/master/src/anglican/csis/network.clj#L111
[edn-link]: https://github.com/edn-format/edn
[png2edn-py-link]: https://github.com/tuananhle7/torch-csis/blob/master/examples/src/helpers/io/png2edn.py
[zmq-rep-req-link]: http://zguide.zeromq.org/page:all#Ask-and-Ye-Shall-Receive
[torch-csis-repo-link]: https://github.com/tuananhle7/torch-csis
[anglican-csis-repo-link]: https://github.com/tuananhle7/anglican-csis
