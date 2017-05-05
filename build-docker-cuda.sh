#/!/bin/bash

INFCOMP_VERSION="$(python setup.py --version)"

docker build -t pytorch-infcomp-cuda -f ./Dockerfile-cuda .
docker tag pytorch-infcomp-cuda pytorch-infcomp-cuda:$INFCOMP_VERSION
