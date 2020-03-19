from typing import Any

from .assembler import Assembler
from .collate import PadCollate
from .data_typing import DataLoaderT, DatasetT
from .glue import GlueDataset
from .mtl_dataset import MtlCatDataset, MtlMixedDataset, MtlSepDataset
from .qa_dataset import QADataset
from .rank_dataset import (
    RankGroupDataset,
    RankPointDataset,
    RankPointSepDataset,
    TokenCountDataset,
)
from .sampler import DynamicBatchSampler, FixedBatchSampler
from .seq2seq_dataset import SummarizationDataset
from .tsv_dataset import TsvCollection
from .utils import (
    get_dynamic_gradient_accumulation,
    make_lm_labels,
    make_targets_mlm_inputs,
    make_targets_ntp_inputs,
    mask_difference,
    mask_whole_word,
    mix_attention_mask,
    sample_by_key,
    split_sequence,
    unpack,
)

__all__ = [
    "get_dynamic_gradient_accumulation",
    "make_lm_labels",
    "mask_difference",
    "mask_whole_word",
    "mix_attention_mask",
    "sample_by_key",
    "make_targets_mlm_inputs",
    "make_targets_ntp_inputs",
    "split_sequence",
    "unpack",
    "Assembler",
    "MtlSepDataset",
    "PadCollate",
    "RankGroupDataset",
    "RankPointDataset",
    "RankPointSepDataset",
    "MtlMixedDataset",
    "TokenCountDataset",
    "SummarizationDataset",
    "DynamicBatchSampler",
    "FixedBatchSampler",
    "TsvCollection",
    "MtlMixedDataset",
    "MtlCatDataset",
    "DataLoaderT",
    "DatasetT",
    "QADataset",
    "get_train_dataset_cls",
    "GlueDataset",
]


def get_dataset_cls(name: str) -> Any:
    return globals()[name]
