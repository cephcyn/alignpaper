#!/bin/bash

# Enter conda environment, reference:
# https://github.com/conda/conda/issues/7980
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate alignpaper

# Partial credits for this script go to
# https://github.com/miguelgrinberg/flask-celery-example

# cd back up to parent directory when leaving script
trap "cd .." EXIT

if [ ! -d redis-stable/src ]; then
    curl -O http://download.redis.io/redis-stable.tar.gz
    tar xvzf redis-stable.tar.gz
    rm redis-stable.tar.gz
fi
cd redis-stable
make
src/redis-server

