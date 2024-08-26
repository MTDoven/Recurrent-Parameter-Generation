accelerate launch \
  --main_process_port=29602 \
  --num_processes=1 \
  --gpu_ids='1' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  cifar10_resnet50_base.py \
