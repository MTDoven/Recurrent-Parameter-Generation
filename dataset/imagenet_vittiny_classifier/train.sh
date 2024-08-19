#!/bin/bash

start_num=0
end_num=0

for i in $(seq $start_num $num_iterations)
do
  echo "\n\n"
  echo "Running iteration $i"
  python train.py "class$i"
done