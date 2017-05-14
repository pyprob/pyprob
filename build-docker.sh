#/!/bin/bash

INFCOMP_VERSION="$(python setup.py --version)"

docker build -t pytorch-infcomp --build-arg INFCOMP_VERSION=$INFCOMP_VERSION .
docker tag pytorch-infcomp pytorch-infcomp:$INFCOMP_VERSION
