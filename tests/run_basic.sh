#!/usr/bin/env bash
# set -e
# docker pull pyprob/pyprob_cpp
pytest -n auto -x -rA -k "not remote and not inference" --cov=./
