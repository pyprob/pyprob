#/!/bin/bash

PYPROB_VERSION="$(python setup.py --version)"
GIT_COMMIT="$(git rev-parse HEAD)"

docker build -t pyprob --build-arg PYPROB_VERSION=$PYPROB_VERSION --build-arg GIT_COMMIT=$GIT_COMMIT .
docker tag pyprob pyprob:$PYPROB_VERSION
