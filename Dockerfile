FROM pyprob-mini

RUN apt-add-repository ppa:jonathonf/texlive-2016
RUN apt update
RUN apt install -y --fix-missing texlive-full
RUN apt install -y libx11-dev nano tmux graphviz

WORKDIR /workspace
RUN chmod -R a+w /workspace
CMD bash
