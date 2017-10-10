#/!/bin/bash

PYPROB_VERSION="$(python setup.py --version)"
GIT_COMMIT="$(git rev-parse HEAD)"

docker build -f Dockerfile-mini -t pyprob-mini --build-arg PYPROB_VERSION=$PYPROB_VERSION --build-arg GIT_COMMIT=$GIT_COMMIT .
docker tag pyprob-mini pyprob-mini:$PYPROB_VERSION

docker build -f Dockerfile -t pyprob --build-arg PYPROB_VERSION=$PYPROB_VERSION --build-arg GIT_COMMIT=$GIT_COMMIT .
docker tag pyprob pyprob:$PYPROB_VERSION
