#!/bin/bash

# cd back up to parent directory when leaving script
trap "cd .." EXIT

conda activate alignpaper

cd reactinterface/
npm start

