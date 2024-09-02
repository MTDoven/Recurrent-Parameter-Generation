accelerate launch \
  --main_process_port=29611 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='2,4' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  imagenet_vitbase_huge.py \
