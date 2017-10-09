import os
import sys
from setuptools import setup
PACKAGE_NAME = 'pyprob'
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
    name='pyprob',
    version=read_package_variable('__version__'),
    description='PyTorch library for Inference Compilation and Universal Probabilistic Programming',
    author='Tuan-Anh Le and Atilim Gunes Baydin',
    packages=['pyprob', 'infcomp.protocol'],
    install_requires=['torch', 'torchvision', 'termcolor==1.1.0', 'pyzmq==16.0.2', 'flatbuffers==2015.12.22.1', 'visdom==0.1.04', 'matplotlib==2.0.0', 'py-cpuinfo==3.2.0', 'pylatex==1.2.1', 'pydotplus==2.0.2', 'seaborn==0.7.1', 'docker==2.4.2'],
    url='https://github.com/probprog/pyprob',
    classifiers=['Development Status :: 4 - Beta', 'License :: OSI Approved :: MIT License', 'Programming Language :: Python :: 3.5'],
    license='MIT',
    keywords='probabilistic programming deep learning inference compilation',
    scripts=['bin/pyprob-compile', 'bin/pyprob-infer']
)
