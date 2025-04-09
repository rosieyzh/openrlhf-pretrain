from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer
from datatrove.data import DocumentsPipeline
import os
import argparse



def process_math(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    for document in data:
        # TODO: should we add chat template?
        document.text = document.text + '\n\n' + document.metadata['code']
        yield document


DATASET_NAME = "tinygsm"
HF_PATH = "hf://datasets/TinyGSM/TinyGSM"
TOKENIZER = "meta-llama/Llama-2-7b-hf"

N_TASKS_PER_NODE = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
NODES = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
RANK = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


DESTINATION = "<PATH_TO_DATA>"

print(f"Running with {N_TASKS_PER_NODE} tasks per node, {NODES} nodes, and rank {RANK}")


reader = ParquetReader(
    data_folder=f"{HF_PATH}/data",
    glob_pattern="*.parquet",
    text_key="question",
    id_key="id",
    read_metadata=True
)


dist_executor = LocalPipelineExecutor(
    pipeline=[
        reader,
        process_math,
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