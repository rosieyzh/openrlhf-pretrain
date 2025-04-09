# Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining

Accompanying code for experiments in "Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining". We are grateful to the contributors of the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [math-verify](git@github.com:huggingface/Math-Verify.git) and [AI2 OLMo](https://github.com/allenai/OLMo) repositories, from which this code is built on top of.

## Setup
The code is up-to-date with OpenRLHF 0.6.3post1, with `vllm` version set to **0.8.1**.

Clone the repository with submodules:
```
git clone --recurse-submodules git@github.com:rosieyzh/openrlhf-pretrain.git
```
And follow setup instructions as per usual for OpenRLHF and OLMo installation.

## Pretraining
Relevant configs and example SLURM scripts are given in `pretraining`. We also provide scripts for converting various datasets used in the paper to tokens for pretraining in `pretraining/data`. The SLURM file launches the script in `OLMo/scripts/train.py` and can be launched as eg. `sbatch pretraining/scripts/run_multinode.sh`.

## RL Fine-tuning
We use the distributed PPO implementation from OpenRLHF using Ray. We've also written an Expert Iteration script at `openrlhf/cli/train_ei.py` based on their SFT implementation. 
* In `scripts` we have python scripts for running SLURM array jobs to sweep over checkpoints and hyperparameters (eg. `scripts/ppo_sweep.py`, `scripts/ei_sweep.py`) with example configs to reproduce the results in our paper in `configs` (eg. `configs/pretraining_1b_sweeps_ppo_grpo_gsm8k.yaml`).
* The PPO SLURM scripts `scripts/run_ppo_sweep.sh` and `scripts/run_ppo_sweep_1_gpu.sh` launch array jobs using 1 node (assuming 4 GPUs per node) and a single GPU, respectively. Due to complexities with Ray, additional care was required to support execution on a single GPU.

## Inference
In `inference` we provide various scripts for evaluating our models, as well as accompanying sweep files to launch SLURM array jobs in a similar manner as fine-tuning. 
* In `inference/run_inference_all.py`, we provide our script for getting pass@1, pass@k and majority@k results using `math-verify` to parse and verify final answers for correctness. Using the flags `--no_greedy` and `--no_multiple` will turn off getting pass@1 or pass@k/majority@k results respectively.
* In `inference/query_gpt.py` we provide our script for qualitatively analyzing MATH generations from the base and fine-tuned model; this script assumes one generation per response and will use the json file generated from getting pass@1 results from `inference/run_inference_all.py`. The python script `inference/query_gpt_all.py` is analogous but was used for analyzing AIME generations from the base and fine-tuned model, using the json file generated from getting pass@64 results from `inference/run_inference_all.py`.