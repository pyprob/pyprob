#!/usr/bin/env bash
sh ./run_basic.sh

echo "Running inference tests"
python test_inference.py
