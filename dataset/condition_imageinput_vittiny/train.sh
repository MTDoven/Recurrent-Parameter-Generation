#!/bin/bash

start=0
end=9

for i in $(seq $start $end)
do
    power=$((2**i))
    srun -p Gveval-S1 --job-name=train --gres=gpu:1 --ntasks-per-node=1 python train.py class$power
    sleep 1
done