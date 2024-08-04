accelerate launch \
  --main_process_port=29503 \
  --num_processes=1 \
  --gpu_ids='7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  convnext_mamba_8192.py \
