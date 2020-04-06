FROM pyprob/pyprob_cpp:latest

ENV PYTHON_VERSION=3.7
ENV CC=gcc-5
ENV CXX=g++-5

RUN apt-get update && apt-get install -y curl python3 python3-pip python3-gdbm
RUN pip3 install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN ln -s $(which python3) /usr/bin/python
WORKDIR /home
COPY . /home/pyprob

RUN pip3 install ./pyprob[dev]
RUN cd pyprob && sh tests/run_basic.sh

ARG PYPROB_VERSION="unknown"
ARG GIT_COMMIT="unknown"

LABEL project="pyprob"
LABEL url="https://github.com/pyprob/pyprob"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL version=$PYPROB_VERSION
LABEL git_commit=$GIT_COMMIT
