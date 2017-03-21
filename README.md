# Torch library for Inference Compilation and Universal Probabilistic Programming

Code for Inference Compilation and Universal Probabilistic Programming ([main project page][project-page-link]).

This repository contains the [Torch](http://torch.ch/)-based neural network part of the inference compilation scheme. The [Clojure](https://clojure.org/)-based probabilistic programming part is implemented as a [Clojure library][anglican-csis-repo-link], extending [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/). The interaction between these two is facilitated by [ZeroMQ](http://zeromq.org/).

For a walkthrough on how to set up a system to compile inference for a probabilistic program written in Anglican, check out the [tutorial](TUTORIAL.md). Also check out the [examples][examples-link] folder.

For documentation of the Torch side, run `compile.lua`, `infer.lua`, or `artifact-info.lua` with the `--help` flag.

If you use this code in your work, please cite our [paper][paper-link]:
```
@article{le2016inference,
  title = {Inference Compilation and Universal Probabilistic Programming},
  author = {Le, Tuan Anh and Baydin, Atilim Gunes and Wood, Frank},
  journal = {arXiv preprint arXiv:1610.09900},
  year = {2016}
}
```

[project-page-link]: http://tuananhle.co.uk/compiled-inference
[anglican-csis-repo-link]: https://github.com/tuananhle7/anglican-csis
[paper-link]: https://arxiv.org/abs/1610.09900
[examples-link]: https://github.com/tuananhle7/torch-csis/tree/master/examples
