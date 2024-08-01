accelerate launch \
  --main_process_port=29502 \
  --multi_gpu \
  --num_processes 2 \
  --gpu_ids='5,6' \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  resnet18_mamba-8192.py \
