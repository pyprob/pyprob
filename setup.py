from setuptools import setup

setup(
    name='pytorch-infcomp',
    version='0.9.3dev',
    description='PyTorch library for Inference Compilation and Universal Probabilistic Programming',
    author='Tuan-Anh Le and Atilim Gunes Baydin',
    packages=['pytorch-infcomp'],
    install_requires=['torch', 'torchvision', 'termcolor>=1.1.0', 'pyzmq>=16.0.2', 'msgpack-python>=0.4.8', 'visdom>=0.1.02'],
    url='https://github.com/probprog/pytorch-infcomp',
    license='GPLv3'
)
