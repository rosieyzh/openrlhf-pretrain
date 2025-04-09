#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --account=<YOUR ACCOUNT>emalach_lab
#SBATCH --output=%A_%a.log
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=250GB		
#SBATCH --partition=kempner
#SBATCH --spread-job

torchrun --nproc_per_node=4 OLMo/scripts/train.py pretraining/configs/RL-150M.yaml