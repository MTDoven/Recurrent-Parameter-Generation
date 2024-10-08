#!/usr/bin/env bash

source /path/to/miniconda3/bin/activate /path/to/miniconda3/envs/environment

python ./convert.py "$1"

PYTHONPATH=/path/to/Segmentation:$PYTHONPATH \
    python /path/to/Segmentation/tools/test.py \
    /path/to/Segmentation/configs/beit/upernet/our_vit.py \
    "$1" \
    --launcher none \
    --eval "mIoU"

python ./reverse.py "$1"

