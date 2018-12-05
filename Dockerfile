FROM ubuntu:18.04

ENV PYTHON_VERSION=3.7
ENV CC=gcc-5
ENV CXX=g++-5

RUN apt-get update && apt-get install -y --no-install-recommends \
        libzmq3-dev \
        git \
        cmake \
        gcc-5 \
        g++-5 \
        curl \
        python3-gdbm \
        ca-certificates &&\
    rm -rf /var/lib/apt/lists/*

RUN git clone --branch v1.10.0 https://github.com/google/flatbuffers.git /code/flatbuffers && cd /code/flatbuffers && cmake -G "Unix Makefiles" && make install
RUN git clone --branch 0.4.16 https://github.com/QuantStack/xtl.git /code/xtl && cd /code/xtl && cmake . && make install
RUN git clone --branch 0.17.4 https://github.com/QuantStack/xtensor.git /code/xtensor && cd /code/xtensor && cmake . && make install
RUN git clone --branch v0.1.7 https://github.com/probprog/pyprob_cpp.git /code/pyprob_cpp && cd /code/pyprob_cpp && mkdir build && cd build && cmake ../src && cmake --build . && make install

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Enable dbm.gnu
RUN cp /usr/lib/python3.7/lib-dynload/_gdbm.cpython-37m-x86_64-linux-gnu.so /opt/conda/lib/python3.7/lib-dynload/

RUN conda install pytorch-nightly-cpu=1.0.0.dev20181205 -c pytorch

RUN mkdir -p /code/pyprob
COPY . /code/pyprob
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
