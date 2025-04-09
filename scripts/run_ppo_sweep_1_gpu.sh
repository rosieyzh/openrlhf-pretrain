#!/bin/bash
#SBATCH --job-name=ppo_gsm8k
#SBATCH --account=<YOUR ACCOUNT>
#SBATCH --output=<YOUR LOGS>/%A_%a.log
#SBATCH --export=ALL
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=150GB
#SBATCH --partition=<YOUR PARTITION>
#SBATCH --constraint=h100
#SBATCH --array=0-89%24

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export GPUS_PER_NODE=1
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( 6000 + $SLURM_ARRAY_TASK_ID ))

# Custom environment
source ~/.bashrc
conda deactivate
conda activate openrlhf

module load cuda
module load cudnn
module load gcc/12.2.0-fasrc01

# Create unique directories for Ray temp files
RAY_TEMP_DIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $RAY_TEMP_DIR

# Function to check if port is available
check_port_available() {
  local port=$1
  # If nc command connects successfully, port is in use
  if nc -z 127.0.0.1 $port >/dev/null 2>&1; then
    # Port is in use
    return 1
  else
    # Port is available
    return 0
  fi
}

# Function to find available port base with more compact spacing
find_available_port_base() {
  local initial_port_base=$1
  local current_port_base=$initial_port_base
  local spacing=$2
  local retries=0
  local max_retries=4
  local found_port=false
  
  # First port to check is the RAY_PORT
  local port_to_check=$current_port_base
  
  while [[ $retries -lt $max_retries && $found_port == false ]]; do
    echo "Checking if port base $current_port_base is available (retry $retries)..."
    
    if check_port_available $port_to_check; then
      echo "Port base $current_port_base is available!"
      PORT_BASE=$current_port_base
      found_port=true
      # Break out of the loop
      break
    else
      echo "Port base $current_port_base is in use, trying next option..."
      retries=$((retries + 1))
      current_port_base=$((current_port_base + spacing))
      port_to_check=$current_port_base
      
      # Ensure we're still in valid range
      if [[ $current_port_base -gt 60000 ]]; then
        echo "Warning: Port base too high during retry, switching to alternative range"
        current_port_base=$((20000 + (RANDOM % 40) * 300))
        port_to_check=$current_port_base
      fi
    fi
  done
  
  # If we didn't find a port, pick a random port base as last resort
  if [[ $found_port == false ]]; then
    echo "Warning: Could not find available port after $max_retries retries, selecting random port base"
    PORT_BASE=$((20000 + (RANDOM % 80) * 200))
  fi
  
  echo "Final selected port base: $PORT_BASE"
}

# Calculate base port with smaller spacing
PORT_SPACING=1100
PORT_BASE=$((10000 + (SLURM_ARRAY_TASK_ID % 40) * PORT_SPACING))

# Try to find an available port base
find_available_port_base $PORT_BASE $PORT_SPACING

# Assign all Ray ports with compact offsets
export RAY_PORT=$PORT_BASE                    # Main Redis port (6379 by default)
export GCS_PORT=$((PORT_BASE + 10))           # GCS Server port
export DASHBOARD_PORT=$((PORT_BASE + 20))     # Dashboard web UI
export NODE_MANAGER_PORT=$((PORT_BASE + 30))  # Node manager
export OBJECT_MANAGER_PORT=$((PORT_BASE + 40)) # Object manager
export DASHBOARD_GRPC_PORT=$((PORT_BASE + 50))        # Dashboard GRPC
export CLIENT_PORT=$((PORT_BASE + 60))        # Ray client server
export DASHBOARD_AGENT_PORT=$((PORT_BASE + 70)) # Dashboard agent
export DASHBOARD_AGENT_LISTEN_PORT=$((PORT_BASE + 80)) # Dashboard agent listener
export METRICS_EXPORT_PORT=$((PORT_BASE + 90)) # Metrics exporter
export RUNTIME_ENV_AGENT_PORT=$((PORT_BASE + 100)) # Metrics agent

# Worker port ranges - keep them smaller but still separate
WORKER_RANGE_SIZE=500
MIN_WORKER_PORT=$((PORT_BASE + 200))
MAX_WORKER_PORT=$((MIN_WORKER_PORT + WORKER_RANGE_SIZE - 1))

# Final safety check for worker ports
if [[ $MAX_WORKER_PORT -gt 65530 ]]; then
  echo "Warning: Worker ports exceed maximum allowed port (65535), adjusting..."
  MIN_WORKER_PORT=$((PORT_BASE + 200))
  MAX_WORKER_PORT=$((MIN_WORKER_PORT + 100)) # Smaller range as fallback
fi

# Print port information for debugging
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Using port base: $PORT_BASE"
echo "Main Ray ports:"
echo "- Redis port: $RAY_PORT"
echo "- GCS port: $GCS_PORT"
echo "- Dashboard port: $DASHBOARD_PORT"
echo "- Node manager port: $NODE_MANAGER_PORT"
echo "- Object manager port: $OBJECT_MANAGER_PORT"
echo "- Dashboard GRPC port: $DASHBOARD_GRPC_PORT"
echo "- Client port: $CLIENT_PORT"
echo "- Dashboard agent port: $DASHBOARD_AGENT_PORT"
echo "- Dashboard agent listen port: $DASHBOARD_AGENT_LISTEN_PORT"
echo "- Metrics export port: $METRICS_EXPORT_PORT"
echo "- Runtime env agent port: $RUNTIME_ENV_AGENT_PORT"
echo "- Worker port range: $MIN_WORKER_PORT-$MAX_WORKER_PORT"

# Start Ray head node with all unique ports and session name
ray start --head \
  --node-ip-address=0.0.0.0 \
  --port=$RAY_PORT \
  --dashboard-port=$DASHBOARD_PORT \
  --node-manager-port=$NODE_MANAGER_PORT \
  --object-manager-port=$OBJECT_MANAGER_PORT \
  --ray-client-server-port=$CLIENT_PORT \
  --dashboard-agent-grpc-port=$DASHBOARD_AGENT_PORT \
  --dashboard-agent-listen-port=$DASHBOARD_AGENT_LISTEN_PORT \
  --dashboard-grpc-port=$DASHBOARD_GRPC_PORT \
  --metrics-export-port=$METRICS_EXPORT_PORT \
  --runtime-env-agent-port=$RUNTIME_ENV_AGENT_PORT \
  --min-worker-port=$MIN_WORKER_PORT \
  --max-worker-port=$MAX_WORKER_PORT \
  --num-gpus=$GPUS_PER_NODE \
  --temp-dir=$RAY_TEMP_DIR \
  --include-dashboard=true

# Add a delay to ensure Ray is fully initialized
echo "Waiting for Ray to initialize..."
sleep 30

# Run the sweep
export SWEEP_CONFIG=configs/pretraining_150m_sweeps_ppo_gsm8k.yaml
python scripts/ppo_sweep.py sweep_config=${SWEEP_CONFIG}

rm -rf $RAY_TEMP_DIR
