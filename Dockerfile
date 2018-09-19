FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV PYTHON_VERSION=3.6
ENV CC=gcc-5
ENV CXX=g++-5

RUN apt-get update && apt-get install -y --no-install-recommends \
        libzmq3-dev \
        git \
        cmake \
        gcc-5 \
        g++-5 \
        curl \
        ca-certificates &&\
    rm -rf /var/lib/apt/lists/*

RUN git clone --branch v1.9.0 https://github.com/google/flatbuffers.git /code/flatbuffers && cd /code/flatbuffers && cmake -G "Unix Makefiles" && make install
RUN git clone --branch 0.4.0 https://github.com/QuantStack/xtl.git /code/xtl && cd /code/xtl && cmake . && make install
RUN git clone --branch 0.15.4 https://github.com/QuantStack/xtensor.git /code/xtensor && cd /code/xtensor && cmake . && make install
RUN git clone --branch v0.1.6 https://github.com/probprog/pyprob_cpp.git /code/pyprob_cpp && cd /code/pyprob_cpp && mkdir build && cd build && cmake ../src && cmake --build . && make install

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p /code/pyprob
COPY . /code/pyprob

RUN pip install 'torch==0.4.1.post2' 'torchvision==0.2.1'
RUN pip install /code/pyprob

ARG PYPROB_VERSION="unknown"
ARG GIT_COMMIT="unknown"

LABEL project="pyprob (with pyprob_cpp)"
LABEL url="https://github.com/probprog/pyprob"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL version=$PYPROB_VERSION
LABEL git_commit=$GIT_COMMIT

WORKDIR /workspace
RUN chmod -R a+w /workspace
