# PyTorch library for Inference Compilation and Universal Probabilistic Programming

Code for Inference Compilation and Universal Probabilistic Programming ([main project page][project-page-link]).

This repository contains the [PyTorch](http://pytorch.org/)-based neural network part of the inference compilation scheme. The [Clojure](https://clojure.org/)-based probabilistic programming part is implemented as a [Clojure library][anglican-csis-repo-link], extending [Anglican](http://www.robots.ox.ac.uk/~fwood/anglican/). The interaction between these two is facilitated by [ZeroMQ](http://zeromq.org/).

For a walkthrough on how to set up a system to compile inference for a probabilistic program written in Anglican, check out the [tutorial](TUTORIAL.md). Also check out the [examples][examples-link] folder.

For documentation of the PyTorch side, run `compile.py`, `infer.py`, or `artifact-info.py` with the `--help` flag.

If you use this code in your work, please cite our [paper][paper-link]:
```
@inproceedings{le2016inference,
  author = {Le, Tuan Anh and Baydin, Atılım Güneş and Wood, Frank},
  booktitle = {20th International Conference on Artificial Intelligence and Statistics, April 20--22, 2017, Fort Lauderdale, FL, USA},
  title = {Inference Compilation and Universal Probabilistic Programming},
  year = {2017}
}
```

## License

Pytorch-infcomp is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Pytorch-infcomp is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Pytorch-infcomp.  If not, see <http://www.gnu.org/licenses/>.    
    
[project-page-link]: http://tuananhle.co.uk/compiled-inference
[anglican-csis-repo-link]: https://github.com/probprog/anglican-inference-compilation
[paper-link]: https://arxiv.org/abs/1610.09900
[examples-link]: https://github.com/probprog/torch-inference-compilation/tree/master/examples
