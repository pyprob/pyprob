FROM pyprob-mini

RUN apt install -y software-properties-common python3-software-properties python-software-properties
RUN apt-add-repository ppa:jonathonf/texlive-2016
RUN apt update
RUN apt install -y --fix-missing texlive-full
RUN apt install -y libx11-dev nano tmux graphviz

WORKDIR /workspace
RUN chmod -R a+w /workspace
CMD bash
