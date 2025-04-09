from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

import os
import argparse
import random

def process_math(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    for document in data:
        # TODO: should we add chat template?
        document.text = document.text + '\n\n' + document.metadata['answer']
        yield document

class Repeat(PipelineStep):
    def __init__(self, num_repeat: int):
        super().__init__()
        self.num_repeat = num_repeat

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for document in data:
            for _ in range(self.num_repeat):
                yield document


DATASET_NAME = "gsm8k"
HF_PATH = "hf://datasets/openai/gsm8k"
TOKENIZER = "meta-llama/Llama-2-7b-hf"

N_TASKS_PER_NODE = 1
NODES = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
RANK = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


DESTINATION = "<PATH_TO_DATA>"

print(f"Running with {N_TASKS_PER_NODE} tasks per node, {NODES} nodes, and rank {RANK}")

for split in ["train", "test"]:
    reader = ParquetReader(
        data_folder=f"{HF_PATH}/main",
        glob_pattern=f"{split}*.parquet",
        text_key="question",
        id_key="id",
        read_metadata=True
    )

    repeater = Repeat(100)

    dist_executor = LocalPipelineExecutor(
        pipeline=[
            reader,
            process_math,
            repeater,
            DocumentTokenizer(
                output_folder=f"{DESTINATION}/{DATASET_NAME}-{split}-tokenized",
                tokenizer_name_or_path=TOKENIZER,
                eos_token="</s>",
                shuffle=True,
                seed=0,
            ),
        ],
        tasks=N_TASKS_PER_NODE * NODES,
        workers=-1,
        logging_dir=f"{DESTINATION}/logs/datatrove/{DATASET_NAME}-{split}",
        # local flags
        local_tasks=N_TASKS_PER_NODE,
        local_rank_offset=RANK * N_TASKS_PER_NODE,
        start_method="fork",
    )
    dist_executor.run()