FROM gbaydin/pytorch-cudnnv6

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=UTF-8
ENV PATH="/code/pytorch-infcomp:${PATH}"

RUN apt update
RUN apt install -y libx11-dev locales nano tmux
RUN locale-gen en_US en_US.UTF-8
RUN dpkg-reconfigure locales

ARG INFCOMP_VERSION="unknown"
ARG GIT_COMMIT="unknown"

LABEL project="pytorch-infcomp"
LABEL url="https://github.com/probprog/pytorch-infcomp"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL version=$INFCOMP_VERSION
LABEL git_commit=$GIT_COMMIT

RUN mkdir -p /code/pytorch-infcomp
COPY . /code/pytorch-infcomp

RUN chmod a+x /code/pytorch-infcomp/compile
RUN chmod a+x /code/pytorch-infcomp/infer
RUN chmod a+x /code/pytorch-infcomp/info

RUN pip install -r /code/pytorch-infcomp/requirements.txt
RUN pip install /code/pytorch-infcomp


CMD bash
