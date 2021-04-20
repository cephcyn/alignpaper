#!/bin/bash

conda activate alignpaper

celery -A api.celery worker --loglevel=info

