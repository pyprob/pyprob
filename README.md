# Torch library for Compiled Inference in the Probabilistic Programming System Anglican

This repository contains the Torch files required to perform the compilation stage in the compiled inference scheme. In our setup, probabilistic program definition and inference is done in the [Anglican Probabilistic Programming System](http://www.robots.ox.ac.uk/~fwood/anglican/) and the inference compilation is done in [Torch](http://torch.ch/). The communication between these two is facilitated by [ZeroMQ](http://zeromq.org/).

For a walkthrough on how to set up a system to compile inference for a probabilistic program written in Anglican, check out the [tutorial](TUTORIAL.md).

Check out the [documentation for the Clojure side](http://tuananhle.co.uk/anglican-csis-doc/). For documentation for the Torch side: run `compile.lua` and `infer.lua` with the `--help` flag.

If you use this code in your work, please consider citing our [paper](https://arxiv.org/abs/1610.09900):
```
@article{le2016inference,
  title = {Inference Compilation and Universal Probabilistic Programming},
  author = {Le, Tuan Anh and Baydin, Atilim Gunes and Wood, Frank},
  journal = {arXiv preprint arXiv:1610.09900},
  year = {2016}
}
```
