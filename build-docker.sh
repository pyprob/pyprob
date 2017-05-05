#/!/bin/bash

INFCOMP_VERSION="$(python setup.py --version)"

docker build -t pytorch-infcomp .
docker tag pytorch-infcomp pytorch-infcomp:$INFCOMP_VERSION
