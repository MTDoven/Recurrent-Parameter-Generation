accelerate launch \
  --main_process_port=29500 \
  --multi_gpu \
  --num_processes 2 \
  --gpu_ids='1,2' \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  convnext_mamba-8192.py \
