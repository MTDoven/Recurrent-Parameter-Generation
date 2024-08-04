accelerate launch \
  --main_process_port=29501 \
  --multi_gpu \
  --num_processes=2 \
  --gpu_ids='0,1' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  convnext_mamba_8192.py \
