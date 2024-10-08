#!/usr/bin/env bash


source /path/to/anaconda3/bin/activate /path/to/anaconda3/envs/dora_llama

python torch2safetensors.py $1 ./evaluate_r4_org
cd ..
sh llama_7B_Dora_eval.sh ./finetuned_result/evaluate_r4_org $CUDA_VISIBLE_DEVICES
cd finetuned_result
python concludeclear.py ./evaluate_r4_org
