accelerate launch \
  --main_process_port=29601 \
  --num_processes=1 \
  --gpu_ids='7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  cifar10_resnet18_base.py \
