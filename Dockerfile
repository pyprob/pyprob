FROM gbaydin/pytorch-cudnnv6

ARG INFCOMP_VERSION="unknown"
ARG GIT_COMMIT="unknown"

LABEL project="pytorch-infcomp"
LABEL url="https://github.com/probprog/pytorch-infcomp"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
LABEL version=$INFCOMP_VERSION
LABEL git_commit=$GIT_COMMIT

RUN apt update
RUN apt install -y libx11-dev

RUN mkdir /home/pytorch-infcomp
COPY . /home/pytorch-infcomp

RUN chmod a+x /home/pytorch-infcomp/compile
RUN chmod a+x /home/pytorch-infcomp/infer
RUN chmod a+x /home/pytorch-infcomp/info

RUN pip install -r /home/pytorch-infcomp/requirements.txt
RUN pip install /home/pytorch-infcomp

ENV PATH="/home/pytorch-infcomp:${PATH}"

CMD bash
