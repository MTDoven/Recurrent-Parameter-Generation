accelerate launch \
  --main_process_port=29544 \
  --num_processes=1 \
  --gpu_ids='2' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  lstm_8192.py \
