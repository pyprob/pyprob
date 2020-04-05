#!/usr/bin/env bash
docker pull pyprob/pyprob_cpp
pytest -n auto -x -rA
