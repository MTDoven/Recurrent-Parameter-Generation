#!/bin/bash

num_iterations=1

for i in $(seq 1 $num_iterations)
do
  echo "\n\n"
  echo "Running iteration $i"
  CUDA_VISIBLE_DEVICES=2 python train.py
done