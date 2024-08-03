accelerate launch \
  --main_process_port=29496 \
  --multi_gpu \
  --num_processes 2 \
  --gpu_ids='3,6' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  tinyvit_mamba-8192.py \
