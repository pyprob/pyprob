#/!/bin/bash

INFCOMP_VERSION="$(python -c "import infcomp; print(infcomp.__version__)")"


docker build -t pytorch-infcomp .
docker tag pytorch-infcomp pytorch-infcomp:$INFCOMP_VERSION
