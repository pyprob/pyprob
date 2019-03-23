#!/usr/bin/env bash
set -e

echo "Running state tests"
python test_state.py

echo "Running trace tests"
python test_trace.py

echo "Running distribution tests"
python test_distributions.py

echo "Running model tests"
python test_model.py

echo "Running remote model tests"
python test_model_remote.py

echo "Running diagnostics tests"
python test_diagnostics.py

echo "Running dataset tests"
python test_dataset.py

echo "Running neural network tests"
python test_nn.py

echo "Running neural network training tests"
python test_train.py

echo "Running util tests"
python test_util.py
