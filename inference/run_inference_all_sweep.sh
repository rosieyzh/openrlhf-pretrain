#!/bin/bash
#SBATCH --job-name=eval_baselines  # Job name
#SBATCH --output=<YOUR LOGS>/%A_%a.log
#SBATCH --time=08:00:00  # Max run time (hh:mm:ss)
#SBATCH --cpus-per-task=16  # Number of CPU cores per task
#SBATCH --gpus-per-node=1
#SBATCH --mem=250G          # Memory per task
#SBATCH --partition=<YOUR PARTITION> # Partition (adjust as needed)
#SBATCH --account=<YOUR ACCOUNT>barak_lab  # Partition (adjust as needed)
#SBATCH --array=0-65%24

source ~/.bashrc
conda deactivate
conda activate openrlhf

module load cuda
module load cudnn
module load gcc/12.2.0-fasrc01

export SWEEP_CONFIG=configs/inference/ei_150m.yaml

python inference/run_inference_all_sweep.py sweep_config=${SWEEP_CONFIG}