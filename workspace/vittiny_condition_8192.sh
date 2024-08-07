accelerate launch \
  --main_process_port=29511 \
  --multi_gpu \
  --num_processes=3 \
  --gpu_ids='1,2,7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  vittiny_condition_8192.py \
