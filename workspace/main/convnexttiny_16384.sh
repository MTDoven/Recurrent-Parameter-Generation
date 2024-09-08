accelerate launch \
  --main_process_port=29904 \
  --num_processes=1 \
  --gpu_ids='0' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  convnexttiny_16384.py \
