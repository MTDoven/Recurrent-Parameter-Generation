accelerate launch \
  --main_process_port=29921 \
  --num_processes=1 \
  --gpu_ids='1' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  no_relation.py \
