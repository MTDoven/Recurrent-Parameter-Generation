accelerate launch \
  --main_process_port=29536 \
  --num_processes=1 \
  --gpu_ids='7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  mixlayers_4096.py \
