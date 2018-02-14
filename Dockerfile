FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p /code/pyprob
COPY . /code/pyprob

RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision
RUN pip install /code/pyprob

ARG PYPROB_VERSION="unknown"
ARG GIT_COMMIT="unknown"

LABEL project="pyprob"
LABEL url="https://github.com/probprog/pyprob"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL version=$PYPROB_VERSION
LABEL git_commit=$GIT_COMMIT

WORKDIR /workspace
RUN chmod -R a+w /workspace
