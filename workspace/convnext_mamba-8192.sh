accelerate launch \
  --multi_gpu \
  --num_processes 4 \
  --gpu_ids="1,2,3,4" \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  convnext_mamba-8192.py \
