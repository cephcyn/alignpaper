#!/bin/bash

# Enter conda environment, reference:
# https://github.com/conda/conda/issues/7980
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate alignpaper

celery -A api.celery worker --loglevel=info

