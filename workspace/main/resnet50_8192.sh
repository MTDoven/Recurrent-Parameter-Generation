accelerate launch \
  --main_process_port=29454 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='2,3' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  resnet50_8192.py \
