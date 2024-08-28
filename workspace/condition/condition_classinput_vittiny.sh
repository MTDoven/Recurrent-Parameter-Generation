srun -p Gveval-S1 --job-name=train --gres=gpu:4 --ntasks-per-node=4 accelerate launch \
  --main_process_port=29631 \
  --multi_gpu \
  --num_processes=4 \
  --gpu_ids='0,1,2,3' \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  condition_classinput_vittiny.py \
