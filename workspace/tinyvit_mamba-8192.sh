accelerate launch \
  --main_process_port=29501 \
  --multi_gpu \
  --num_processes 2 \
  --gpu_ids='3,4' \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  tinyvit_mamba-8192.py \
