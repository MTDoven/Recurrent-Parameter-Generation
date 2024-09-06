accelerate launch \
  --main_process_port=29803 \
  --num_processes=1 \
  --gpu_ids='6' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  dim_4096.py \
