accelerate launch \
  --main_process_port=29601 \
  --num_processes=1 \
  --gpu_ids='1' \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  imagenet_resnet18_base.py \
