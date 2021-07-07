#!/bin/bash

# Enter conda environment, reference:
# https://github.com/conda/conda/issues/7980
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate alignpaper

export FLASK_APP=api.py
export FLASK_ENV=development

flask run --no-debugger

