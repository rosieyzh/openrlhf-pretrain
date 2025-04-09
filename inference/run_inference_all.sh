#!/bin/bash
#SBATCH --job-name=eval_baselines  # Job name
#SBATCH --account=<YOUR ACCOUNT>
#SBATCH --output=<YOUR LOGS>/%A.log
#SBATCH --time=08:00:00  # Max run time (hh:mm:ss)
#SBATCH --cpus-per-task=16  # Number of CPU cores per task
#SBATCH --gpus-per-node=1
#SBATCH --mem=250G          # Memory per task
#SBATCH --partition=<YOUR PARTITION> # Partition (adjust as needed)

module load cuda
module load cudnn
module load gcc/12.2.0-fasrc01

source ~/.bashrc
conda deactivate
conda activate openrlhf

CHECKPOINT_PATH='<INSERT PATH TO CHECKPOINT HERE>'

python inference/run_inference_all.py -c $CHECKPOINT_PATH


