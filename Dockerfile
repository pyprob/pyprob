FROM continuumio/anaconda3

LABEL maintainer "gunes@robots.ox.ac.uk"

RUN mkdir /home/pytorch-infcomp
COPY . /home/pytorch-infcomp

RUN chmod a+x /home/pytorch-infcomp/compile
RUN chmod a+x /home/pytorch-infcomp/infer
RUN chmod a+x /home/pytorch-infcomp/info

RUN conda install pytorch torchvision -c soumith

RUN pip install -r /home/pytorch-infcomp/requirements.txt
RUN pip install /home/pytorch-infcomp

ENV PATH="/home/pytorch-infcomp:${PATH}"

CMD bash
