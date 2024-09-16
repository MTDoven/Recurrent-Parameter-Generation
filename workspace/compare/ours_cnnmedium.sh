accelerate launch \
  --main_process_port=29713 \
  --num_processes=1 \
  --gpu_ids='3' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  ours_cnnmedium.py \
