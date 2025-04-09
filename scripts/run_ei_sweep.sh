#!/bin/bash
#SBATCH --job-name=ei_sweep
#SBATCH --account=<YOUR ACCOUNT>
#SBATCH --output=<YOUR LOGS>/%A_%a.log
#SBATCH --export=ALL
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --mem=250GB
#SBATCH --partition=<YOUR PARTITION>
#SBATCH --constraint=h100
#SBATCH --array=1-45%5


export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export GPUS_PER_NODE=1
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT_CANDIDATES=(6000 6001 6002 6003)
# check if master port candidate is already being used; if not, use it as master port
for MPC in ${MASTER_PORT_CANDIDATES[@]}; do
    NUM_LISTENING_PROCESSES=$(lsof -Pi :${MPC} -sTCP:LISTEN | wc -l)
    if test $NUM_LISTENING_PROCESSES -eq 0; then
        MASTER_PORT=${MPC}
        export MASTER_PORT=${MPC}
        echo "Setting master port to ${MASTER_PORT}."
        break
    fi
done
if [ -z ${MASTER_PORT+x} ]; then
    echo "Could not find an available master port. Exiting."
    exit
fi

# Custom environment
source ~/.bashrc
conda deactivate
conda activate openrlhf

module load cuda
module load cudnn
module load gcc/12.2.0-fasrc01

export SWEEP_CONFIG=configs/pretraining_150m_sweeps_ei.yaml

python scripts/ei_sweep.py sweep_config=${SWEEP_CONFIG}
