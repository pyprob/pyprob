# Torch library for Compiled Inference in the Probabilistic Programming System Anglican

## Dependencies
- [Torch](http://torch.ch/)
- [Torch Autograd](https://github.com/twitter/torch-autograd)
- etc. (TODO)

## Usage
- Run `compile.lua` when compiling a query via a specified TCP port.
- Run `infer.lua` when running compiled inference on the same query.

Compiled artifacts are saved to `./data` by default.  
Use `--help` option for help.
