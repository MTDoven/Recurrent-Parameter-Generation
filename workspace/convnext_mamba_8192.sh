accelerate launch \
  --main_process_port=29502 \
  --multi_gpu \
  --num_processes=4 \
  --gpu_ids='1,2,5,7' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  convnext_mamba_8192.py \
