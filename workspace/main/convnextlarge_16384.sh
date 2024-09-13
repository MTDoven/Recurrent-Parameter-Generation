accelerate launch \
  --main_process_port=29712 \
  --multi_gpu \
  --num_processes=4 \
  --gpu_ids='2,3,4,5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  convnextlarge_16384.py \
