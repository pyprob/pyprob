FROM gbaydin/pytorch-cudnnv6

RUN apt update
RUN apt install -y libx11-dev

ARG INFCOMP_VERSION="unknown"
ARG GIT_COMMIT="unknown"

LABEL project="pytorch-infcomp"
LABEL url="https://github.com/probprog/pytorch-infcomp"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL version=$INFCOMP_VERSION
LABEL git_commit=$GIT_COMMIT

RUN mkdir /code/pytorch-infcomp
COPY . /code/pytorch-infcomp

RUN chmod a+x /code/pytorch-infcomp/compile
RUN chmod a+x /code/pytorch-infcomp/infer
RUN chmod a+x /code/pytorch-infcomp/info

RUN pip install -r /code/pytorch-infcomp/requirements.txt
RUN pip install /code/pytorch-infcomp

ENV PATH="/code/pytorch-infcomp:${PATH}"

CMD bash
