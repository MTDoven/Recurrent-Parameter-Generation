#!/bin/bash

start=67
end=99

for i in $(seq $start $end)
do
    srun -p Gveval-S1 --job-name=train --gres=gpu:1 --ntasks-per-node=1 python train.py class$i
    sleep 1
done