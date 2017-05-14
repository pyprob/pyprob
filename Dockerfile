FROM gbaydin/pytorch-cudnnv6

LABEL project="pytorch-infcomp"
LABEL url="https://github.com/probprog/pytorch-infcomp"
LABEL maintainer="Atilim Gunes Baydin <gunes@robots.ox.ac.uk>"
ARG INFCOMP_VERSION=unknown
LABEL version=$INFCOMP_VERSION

RUN mkdir /home/pytorch-infcomp
COPY . /home/pytorch-infcomp

RUN chmod a+x /home/pytorch-infcomp/compile
RUN chmod a+x /home/pytorch-infcomp/infer
RUN chmod a+x /home/pytorch-infcomp/info

RUN pip install -r /home/pytorch-infcomp/requirements.txt
RUN pip install /home/pytorch-infcomp

ENV PATH="/home/pytorch-infcomp:${PATH}"

CMD bash
