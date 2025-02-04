srun -p Gveval-S1 --job-name=train --gres=gpu:3 --ntasks-per-node=1 accelerate launch \
  --main_process_ip='scontrol show hostname $SLURM_JOB_NODELIST | head -n1' \
  --main_process_port=$((RANDOM % 101 + 20000)) \
  --multi_gpu \
  --num_processes=3 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  generalization.py \