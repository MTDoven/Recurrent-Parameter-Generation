accelerate launch \
  --main_process_port=29505 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='3,4' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  resnet18_mamba_8192.py \
