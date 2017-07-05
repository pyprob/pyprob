FROM gbaydin/pytorch

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV PYTHONIOENCODING=UTF-8
ENV PATH="/code/pytorch-infcomp:${PATH}"

RUN rm -fR /var/lib/apt/lists/*
RUN apt update
RUN apt install -y apt-utils locales
RUN apt install -y software-properties-common python3-software-properties python-software-properties
RUN apt-add-repository ppa:jonathonf/texlive-2016
RUN apt update
RUN apt install -y --fix-missing texlive-full
RUN apt install -y libx11-dev nano tmux graphviz

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
RUN chmod a+x /code/pytorch-infcomp/analytics

RUN pip install -r /code/pytorch-infcomp/requirements.txt
RUN pip install /code/pytorch-infcomp

WORKDIR /workspace
RUN chmod -R a+w /workspace
CMD bash
