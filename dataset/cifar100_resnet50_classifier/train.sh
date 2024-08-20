#!/bin/bash

for i in $(seq 0 33)
do
  echo "\n\n"
  echo "Running iteration $i"
  srun -p Gveval-S1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 python train.py class$i
done