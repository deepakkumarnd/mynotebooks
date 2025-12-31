#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Activate virtual environment
source ./venv/bin/activate

# Optional: show which python is used
which python

# Start Jupyter Notebook
jupyter notebook

