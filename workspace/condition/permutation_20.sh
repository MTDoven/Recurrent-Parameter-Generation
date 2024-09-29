srun -p Gveval-S1 --job-name=train --gres=gpu:2 --ntasks-per-node=1 accelerate launch \
  --main_process_ip='scontrol show hostname $SLURM_JOB_NODELIST | head -n1' \
  --main_process_port=$((RANDOM % 101 + 20000)) \
  --multi_gpu \
  --num_processes=2\
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=no \
  permutation_20.py \