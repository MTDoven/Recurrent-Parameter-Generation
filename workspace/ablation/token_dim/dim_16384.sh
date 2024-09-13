accelerate launch \
  --main_process_port=29404 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='4,5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  dim_16384.py \
