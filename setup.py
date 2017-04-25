from setuptools import setup
import os
with open(os.path.join('.', 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name='infcomp',
    version=version,
    description='PyTorch library for Inference Compilation and Universal Probabilistic Programming',
    author='Tuan-Anh Le and Atilim Gunes Baydin',
    packages=['infcomp', 'infcomp.flatbuffers'],
    install_requires=['torch', 'torchvision==0.1.8', 'termcolor==1.1.0', 'pyzmq==16.0.2', 'flatbuffers==2015.12.22.1', 'visdom==0.1.02', 'matplotlib==2.0.0'],
    url='https://github.com/probprog/pytorch-inference-compilation',
    license='GPLv3'
)
