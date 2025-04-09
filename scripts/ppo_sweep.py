"""
Runs a sweep over the specified file.
To use, specify `sweep_config`, `dist_config`, and `script_name` arguments.
"""

import subprocess
from itertools import product
from omegaconf import OmegaConf
import os
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ppo_sweep")

def flatten(config):
    """Flatten a nested dictionary."""
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat_config[f"{k}.{k2}"] = v2
        else:
            flat_config[k] = v
    return flat_config


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    flat_grid = flatten(grid)
    iter_overwrites = {}
    flat_overwrites = {}
    for k, v in flat_grid.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v

    product_values = list(product(*iter_overwrites.values()))
    grid_list = []
    for values in product_values:
        overwrite_dict = dict(zip(iter_overwrites.keys(), values))
        overwrite_dict.update(flat_overwrites)
        grid_list.append(overwrite_dict)
    return grid_list


def run(cli_args):
    if "debug" in cli_args:
        logger.info("Debug Mode")
        master_addr = 5555
        master_port = 6002
        machine_rank = 0
        num_processes = 4
        num_machines = 1

        slurm_cpus_per_task = 4
        slurm_job_id = 0
        slurm_task_id = 1
        dashboard_port = 8265
    else:
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        machine_rank = os.environ.get("SLURM_PROCID")
        num_processes = os.environ.get("NUM_PROCESSES")
        num_machines = os.environ.get("NNODES")

        slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        dashboard_port = int(os.environ.get("DASHBOARD_PORT", 8265))

    logger.info(f"Starting task ID {slurm_task_id} (Job ID: {slurm_job_id})")

    # Compute overrides
    try:
        base_sweep = OmegaConf.load(cli_args.sweep_config)
        list_of_sweeps = base_sweep.pop("sweep")
        config_list = []
        for sweep in list_of_sweeps:
            sweep_config = OmegaConf.merge(base_sweep, sweep)
            config_list += grid_to_list(sweep_config)

        if slurm_task_id >= len(config_list):
            logger.error(f"Task ID {slurm_task_id} exceeds the number of configurations {len(config_list)}")
            return
            
        overrides = config_list[slurm_task_id]

        new_path = ["ppo"]
        for key in list(overrides.keys()):
            if key in base_sweep.keys(): continue
            if key == 'pretrain':
                if 'save_path' in overrides:
                    overrides['save_path'] = os.path.join(overrides['save_path'], str(overrides[key]))
                else:
                    overrides['save_path'] = str(overrides[key])
                continue
            if key in ['use_kl_loss', 'kl_estimator']:
                continue
            new_path += [key]
            new_path += [str(overrides[key])]
        
        # Create save path directory
        if 'MATH' in overrides['prompt_data']:
            overrides['save_path'] = os.path.join(overrides['save_path'], '_'.join(new_path + ['math']))
        else:
            overrides['save_path'] = os.path.join(overrides['save_path'], '_'.join(new_path))
        overrides['ckpt_path'] = os.path.join(overrides['save_path'], 'ckpt')
        os.makedirs(overrides['save_path'], exist_ok=True)
        os.makedirs(overrides['ckpt_path'], exist_ok=True)
        
        logger.info(f'Save path: {overrides["save_path"]}')
        logger.info(f'Checkpoint path: {overrides["ckpt_path"]}')
        logger.info(f"Configuration for task {slurm_task_id}: dashboard port {dashboard_port}")
        logger.info(overrides)
        
        if "debug" in cli_args:
            logger.info(f"Total configs: {len(config_list)}")
            logger.info(config_list)
        
        # Ray job submission command
        launch_args = [
            f"ray job submit --address='http://127.0.0.1:{dashboard_port}'",
            '--runtime-env-json=\'{\"working_dir\":\"./openrlhf_work_dir\"}\'',
            '-- python3 -m openrlhf.cli.train_ppo_ray',
            f"--packing_samples",
            f"--bf16",
            f"--flash_attn",
            f"--gradient_checkpointing",
            f"--adam_offload",
            f"--normalize_reward",
            f"--save_hf_ckpt"
        ]

        no_chat_template = False
        if 'no_chat_template' in overrides:
            no_chat_template = overrides.pop('no_chat_template')

        if not no_chat_template:
            launch_args.append(f"--apply_chat_template")
        
        disable_ds_ckpt = False
        if 'disable_ds_ckpt' in overrides:
            disable_ds_ckpt = overrides.pop('disable_ds_ckpt')

        if disable_ds_ckpt:
            launch_args.append(f"--disable_ds_ckpt")
        
        colocate_all_models = False
        if 'colocate_all_models' in overrides:
            colocate_all_models = overrides.pop('colocate_all_models')

        if colocate_all_models:
            launch_args.append(f"--colocate_all_models")
            launch_args.append(f"--enable_prefix_caching")
            launch_args.append(f"--vllm_enable_sleep")

        for k, v in overrides.items():
            if k == "use_kl_loss":
                if v:
                    launch_args.append("--use_kl_loss")
            elif k == "clip":
                if not v:
                    launch_args.append("--no_clip")
            elif k == "importance_sampling":
                if not v:
                    launch_args.append("--no_importance_sampling")
            else:
                launch_args.append(f"--{k}={v}")

        launch_args.append(f"--slurm_job={slurm_job_id}_{slurm_task_id}")

        if "debug" in cli_args:
            logger.info(f"Launch command: {' '.join(launch_args)}")

        # Maximum retries for Ray connection
        max_retries = 3
        retry_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Submitting job, attempt {attempt+1}/{max_retries}")
                result = subprocess.run([
                    "bash",
                    "-c", ' '.join(launch_args)
                ], capture_output=True, text=True, check=False)
                
                if result.returncode != 0:
                    logger.error(f"Command failed with return code {result.returncode}")
                    logger.error(f"STDOUT: {result.stdout}")
                    logger.error(f"STDERR: {result.stderr}")
                    
                    # Check for specific error patterns
                    if "No available agent" in result.stderr or "500" in result.stderr:
                        logger.warning("Ray agent not available, retrying...")
                        time.sleep(retry_delay)
                        continue
                else:
                    logger.info("Job submitted successfully")
                    break
            except Exception as e:
                logger.exception(f"Error running command: {e}")
                time.sleep(retry_delay)
        else:
            logger.error(f"Failed to submit job after {max_retries} attempts")
            
    except Exception as e:
        logger.exception(f"Error in run function: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    run(cli_args)