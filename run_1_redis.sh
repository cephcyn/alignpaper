#!/bin/bash

# Partial credits for this script go to
# https://github.com/miguelgrinberg/flask-celery-example

# cd back up to parent directory when leaving script
trap "cd .." EXIT

conda activate alignpaper

if [ ! -d redis-stable/src ]; then
    curl -O http://download.redis.io/redis-stable.tar.gz
    tar xvzf redis-stable.tar.gz
    rm redis-stable.tar.gz
fi
cd redis-stable
make
src/redis-server

