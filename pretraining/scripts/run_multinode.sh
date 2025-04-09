#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --account=<YOUR ACCOUNT>emalach_lab
#SBATCH --output=%A_%a.log
#SBATCH --nodes=2              # Updated to 2 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=1000GB		
#SBATCH --partition=<YOUR PARTITION>
#SBATCH --exclude=holygpu8a15501

export WANDB_API_KEY=
export PYTHONPATH="${PYTHONPATH}:/"
export WANDB_DATA_DIR=

# these are just debugging flags, can disable
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export NCCL_DEBUG=INFO


export CUDA_DEVICE_MAX_CONNECTIONS=1 # Important for Nanotron
# export OMP_NUM_THREADS=16

# EDIT if it's not 8-gpus per node
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
NNODES=$SLURM_NNODES

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=25678
echo "Master addr: $MASTER_ADDR:$MASTER_PORT"

LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $SLURM_NODEID \
    --rdzv_id $SLURM_JOBID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role $(hostname -s|tr -dc '0-9'): \
    --tee 3 \
"

# Check that relative paths to your `run_train.py` are correct
PROGRAM="OLMo/scripts/train.py pretraining/configs/RL-1B.yaml"

export CMD="${LAUNCHER} ${PROGRAM}"

echo $CMD

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

srun $SRUN_ARGS bash -c "$CMD" 2>&1 # Run training across both nodes (8 GPUs total)