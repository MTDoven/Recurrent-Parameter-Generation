accelerate launch \
  --main_process_port=29711 \
  --multi_gpu \
  --num_processes=3 \
  --gpu_ids='1,2,5' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  ours_mobilenetv3.py \
