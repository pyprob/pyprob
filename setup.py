import os
import sys
from setuptools import setup
PACKAGE_NAME = 'infcomp'
MINIMUM_PYTHON_VERSION = 3, 5

def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))

def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)

check_python_version()
setup(
    name='infcomp',
    version=read_package_variable('__version__'),
    description='PyTorch library for Inference Compilation and Universal Probabilistic Programming',
    author='Tuan-Anh Le and Atilim Gunes Baydin',
    packages=['infcomp', 'infcomp.flatbuffers'],
    install_requires=['torch', 'torchvision==0.1.8', 'termcolor==1.1.0', 'pyzmq==16.0.2', 'flatbuffers==2015.12.22.1', 'visdom==0.1.02', 'matplotlib==2.0.0'],
    url='https://github.com/probprog/pytorch-inference-compilation',
    license='GPLv3'
)
