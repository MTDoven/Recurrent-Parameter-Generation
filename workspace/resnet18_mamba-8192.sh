accelerate launch \
  --main_process_port=29502 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='5,7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  resnet18_mamba-8192.py \
