#/!/bin/bash

INFCOMP_VERSION="$(python setup.py --version)"
GIT_COMMIT="$(git rev-parse HEAD)"

docker build -t pytorch-infcomp --build-arg INFCOMP_VERSION=$INFCOMP_VERSION --build-arg GIT_COMMIT=$GIT_COMMIT .
docker tag pytorch-infcomp pytorch-infcomp:$INFCOMP_VERSION
