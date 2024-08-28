accelerate launch \
  --main_process_port=29601 \
  --num_processes=1 \
  --gpu_ids='5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  imagenet_resnet18_large.py \
