accelerate launch \
  --main_process_port=29601 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='0,1' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  cifar10_resnet18_base.py \
