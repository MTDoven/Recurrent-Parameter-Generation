accelerate launch \
  --main_process_port=29703 \
  --num_processes=1 \
  --gpu_ids='0' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  resnet50_8192.py \
