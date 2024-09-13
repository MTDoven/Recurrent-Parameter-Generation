accelerate launch \
  --main_process_port=29804 \
  --num_processes=1 \
  --gpu_ids='5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  dim_1024.py \
