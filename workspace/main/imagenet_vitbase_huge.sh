accelerate launch \
  --main_process_port=29611 \
  --num_processes=1 \
  --gpu_ids='7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  imagenet_vitbase_huge.py \
