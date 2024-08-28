accelerate launch \
  --main_process_port=29602 \
  --num_processes=1 \
  --gpu_ids='7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  imagenet_resnet50_large.py \
