from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer
import os
import argparse


DATASET_NAME = 'algebraic-stack'
HF_PATH = "hf://datasets/EleutherAI/proof-pile-2"
TOKENIZER = "meta-llama/Llama-2-7b-hf"

N_TASKS_PER_NODE = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
NODES = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
RANK = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

DESTINATION = "<PATH_TO_DATA>"

print(f"Running with {N_TASKS_PER_NODE} tasks per node, {NODES} nodes, and rank {RANK}")


def default_adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": [],
        "metadata": {},
    }


dist_executor = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(
            HF_PATH,  # read directly from huggingface
            glob_pattern=f"{DATASET_NAME}/train/*.jsonl.zst",
            compression="zstd",
            text_key="text",
            id_key="id",
            adapter=default_adapter,
        ),
        DocumentTokenizer(
            output_folder=f"{DESTINATION}/{DATASET_NAME}-tokenized",
            tokenizer_name_or_path=TOKENIZER,
            eos_token="</s>",
            shuffle=True,
            seed=0,
        ),
    ],
    tasks=N_TASKS_PER_NODE * NODES,
    workers=-1,
    logging_dir=f"{DESTINATION}/logs/datatrove/{DATASET_NAME}",
    # local flags
    local_tasks=N_TASKS_PER_NODE,
    local_rank_offset=RANK * N_TASKS_PER_NODE,
    start_method="fork",
)
dist_executor.run()