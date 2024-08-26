#!/bin/bash

start=1
end=100

for ((i=start; i<=end; i++))
do
    CUDA_VISIBLE_DEVICES=0 python train.sh class$i
done