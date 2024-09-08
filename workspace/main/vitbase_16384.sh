accelerate launch \
  --main_process_port=29902 \
  --multi_gpu \
  --num_processes=4 \
  --gpu_ids='2,3,4,6' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  vitbase_16384.py \
