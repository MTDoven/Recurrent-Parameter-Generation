#!/bin/bash

num_iterations=9

for i in $(seq 1 $num_iterations)
do
  echo "\n\n"
  echo "Running iteration $i"
  CUDA_VISIBLE_DEVICES=3 python train.py $i
done