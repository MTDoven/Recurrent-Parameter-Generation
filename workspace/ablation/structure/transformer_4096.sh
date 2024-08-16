accelerate launch \
  --main_process_port=29541 \
  --num_processes=1 \
  --gpu_ids='2' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  transformer_4096.py \
