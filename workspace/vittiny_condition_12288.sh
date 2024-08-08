accelerate launch \
  --main_process_port=29515 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='5,7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  vittiny_condition_12288.py \
