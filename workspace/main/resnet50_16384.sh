accelerate launch \
  --main_process_port=29909 \
  --num_processes=1 \
  --gpu_ids='5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  resnet50_16384.py \
