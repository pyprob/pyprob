# Inference Compilation and Universal Probabilistic Programming

Probabilistic inference is hard. *Compile inference* for probabilistic programs written in [Anglican][anglican-link] to obtain a *compilation artifact*, speeding up probabilistic inference during runtime.

**Compilation** of probabilistic programs refers to training a neural network for constructing highly efficient proposal distributions for importance sampling based inference algorithms. The **compilation artifact** consists of an automatically-built program-specific neural architecture and trained weights. During the inference **runtime**, sequential importance sampling is run with proposal distributions obtained from the compilation artifact.

The neural network part of this scheme is written in [Torch][torch-link]. The probabilistic programming part is written in [Clojure][clojure-link] as a library, extending [Anglican][anglican-link]. The communication between the two is facilitated by [ZeroMQ][zeromq-link].

To get started, please visit:
- [Tutorial][tutorial-link]: A complete walkthrough
- [Examples][examples-link]: Examples for reproducing experiments in the [paper][paper-link]

For the code, please visit:
- [Torch code][torch-csis-repo-link]: Torch-based neural network part
- [Clojure code][anglican-csis-repo-link]: Clojure-based library for the probabilistic programming part
- [Leiningen template code][anglican-csis-template-repo-link]: Ready-to-go [Leiningen][leiningen-link] template for the Clojure part

For more details, please check out our [paper][paper-link].
```
@article{le2016inference,
  title = {Inference Compilation and Universal Probabilistic Programming},
  author = {Le, Tuan Anh and Baydin, Atilim Gunes and Wood, Frank},
  journal = {arXiv preprint arXiv:1610.09900},
  year = {2016}
}
```

[torch-csis-repo-link]: https://github.com/tuananhle7/torch-csis
[anglican-csis-repo-link]: https://github.com/tuananhle7/anglican-csis
[anglican-csis-template-repo-link]: https://github.com/tuananhle7/anglican-csis-template
[tutorial-link]: https://github.com/tuananhle7/torch-csis/blob/master/TUTORIAL.md
[examples-link]: https://github.com/tuananhle7/torch-csis/tree/master/examples
[paper-link]: https://arxiv.org/abs/1610.09900
[anglican-link]: http://www.robots.ox.ac.uk/~fwood/anglican
[torch-link]: http://torch.ch/
[clojure-link]: https://clojure.org/
[zeromq-link]: http://zeromq.org/
[leiningen-link]: http://leiningen.org/
