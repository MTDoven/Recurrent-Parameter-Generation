accelerate launch \
  --main_process_port=29498 \
  --multi_gpu \
  --num_processes 2 \
  --gpu_ids='1,4' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  convnext_mamba-8192.py \
