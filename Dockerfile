FROM gbaydin/pytorch-cudnnv6

RUN mkdir /home/pytorch-infcomp
COPY . /home/pytorch-infcomp

RUN pip install -r /home/pytorch-infcomp/requirements.txt
RUN pip install /home/pytorch-infcomp

RUN printf "#!/bin/bash\n python -m infcomp.compile $@" >> /usr/local/bin/compile
RUN chmod a+x /usr/local/bin/compile

RUN printf "#!/bin/bash\n $@" >> /usr/local/bin/run
RUN chmod a+x /usr/local/bin/run

ENTRYPOINT ["/usr/local/bin/run"]
