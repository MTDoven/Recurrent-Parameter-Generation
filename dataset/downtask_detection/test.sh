#!/usr/bin/env bash

source /path/to/miniconda3/bin/activate /path/to/miniconda3/envs/environment

CLUSTER=True \
DETECTRON2_DATASETS="/path/to/" \
PYTHONPATH="$(dirname $0)/Detection":$PYTHONPATH \
python $(dirname $0)/Detection/tools/lazyconfig_train_net.py --config-file $(dirname $0)/Detection/projects/ViTDet/configs/COCO/our_vit_b_100ep.py --finetune "VIT_BASE_IN21K" \
--num-gpus 1 \
--fulltune \
--eval-only "train.init_checkpoint='$1'"
