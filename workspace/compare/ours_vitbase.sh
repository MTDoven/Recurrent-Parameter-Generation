accelerate launch \
  --main_process_port=29719 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='1,5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  ours_vitbase.py \
