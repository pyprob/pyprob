FROM gbaydin/pytorch-cudnnv6

RUN mkdir /home/pytorch-infcomp
COPY . /home/pytorch-infcomp

RUN pip install -r /home/pytorch-infcomp/requirements.txt
RUN pip install /home/pytorch-infcomp
