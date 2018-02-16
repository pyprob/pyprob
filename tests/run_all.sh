#!/usr/bin/env bash
set -e

echo "Running state tests"
python test_state.py

echo "Running distribution tests"
python test_distributions.py

echo "Running model tests"
python test_model.py

echo "Running inference tests"
python test_inference.py
