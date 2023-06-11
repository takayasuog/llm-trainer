import datasets
from datasets import (
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
)
from typing import Union

from configs.finetune_config import FinetuneConfig

def load_dataset(conf: FinetuneConfig) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    if conf.data_path.endswith(".json") or conf.data_path.endswith(".jsonl"):
        data = datasets.load_dataset("json", data_files=conf.data_path)
    else:
        data = datasets.load_dataset(conf.data_path)
    
    return data


