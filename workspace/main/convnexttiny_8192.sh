accelerate launch \
  --main_process_port=29502 \
  --num_processes=1 \
  --gpu_ids='4' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  convnexttiny_8192.py \
