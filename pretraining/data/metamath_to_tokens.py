from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer
from datatrove.data import DocumentsPipeline
import os
import argparse

from typing import Callable, Literal

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.utils.logging import logger


class JsonlMultilineReader(BaseDiskReader):
    """Read data from JSONL files.
        Will read each line as a separate document.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        compression: the compression to use (default: "infer")
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    name = "🐿 Jsonl"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        self.compression = compression

    def read_file(self, filepath: str):
        import orjson
        from orjson import JSONDecodeError

        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                for li, line in enumerate(f):
                    with self.track_time():
                        try:
                            json = orjson.loads(line)
                            for ei, entry in enumerate(json):
                                document = self.get_document_from_dict(entry, filepath, ei)
                                if not document:
                                    continue
                                yield document
                        except (EOFError, JSONDecodeError) as e:
                            logger.warning(f"Error when reading `{filepath}`: {e}")
                            continue
                    
            except UnicodeDecodeError as e:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")

def adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "id":  id_in_file,
        "metadata": data,
        "text": data["query"]
    }

def process_math(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    for document in data:
        # TODO: should we add chat template?
        document.text = document.text + '\n\n' + document.metadata['response']
        yield document


DATASET_NAME = "MetaMathQA"
HF_PATH = "hf://datasets/meta-math/MetaMathQA"
TOKENIZER = "meta-llama/Llama-2-7b-hf"

N_TASKS_PER_NODE = 1
NODES = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
RANK = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


DESTINATION = "<PATH_TO_DATA>"

print(f"Running with {N_TASKS_PER_NODE} tasks per node, {NODES} nodes, and rank {RANK}")

reader = JsonlMultilineReader(
            HF_PATH,  # read directly from huggingface
            glob_pattern="MetaMathQA-395K.json",
            text_key="query",
            id_key="id",
            adapter=adapter
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