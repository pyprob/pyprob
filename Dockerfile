FROM pytorch/pytorch

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV PYTHONIOENCODING=UTF-8
ENV PATH="/code/pyprob:${PATH}"

RUN rm -fR /var/lib/apt/lists/*
RUN apt update
RUN apt install -y apt-utils locales
RUN apt install -y software-properties-common python3-software-properties python-software-properties
RUN apt-add-repository ppa:jonathonf/texlive-2016
RUN apt update
RUN apt install -y --fix-missing texlive-full
RUN apt install -y libx11-dev nano tmux graphviz

ARG PYPROB_VERSION="unknown"
ARG GIT_COMMIT="unknown"

LABEL project="pyprob"
LABEL url="https://github.com/probprog/pyprob"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL version=$PYPROB_VERSION
LABEL git_commit=$GIT_COMMIT

RUN mkdir -p /code/pyprob
COPY . /code/pyprob

RUN chmod a+x /code/pyprob/compile
RUN chmod a+x /code/pyprob/infer
RUN chmod a+x /code/pyprob/analytics

RUN pip install -r /code/pyprob/requirements.txt
RUN pip install /code/pyprob

WORKDIR /workspace
RUN chmod -R a+w /workspace
CMD bash
