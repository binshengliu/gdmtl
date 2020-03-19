from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from .assembler import Assembler
from .data_typing import DatasetT
from .tsv_dataset import TsvCollection

log = logging.getLogger(__name__)


class SummarizationDataset(DatasetT):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sum_prefix: str,
        decoder_start_token_id: int,
        data: str,
        src_max_length: int,
        tgt_max_length: Optional[int] = None,
        pad_to_max_length: bool = True,
        sort: Optional[str] = None,
    ):
        assert sort in [None, "ascending", "descending"]

        super(SummarizationDataset, self).__init__()
        self._tokenizer = tokenizer
        self._decoder_start_token_id = decoder_start_token_id

        self._data = TsvCollection(data)
        self._indexes = np.array(list(self._data.keys()))

        if sort is not None:
            lengths = np.array([len(self._data[i].split()) for i in self._indexes])
            lengths = lengths if sort == "ascending" else -lengths
            sorted_indexes = np.argsort(lengths)
            self._indexes = self._indexes[sorted_indexes]

        self._src_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=sum_prefix,
            pad_to_max_length=pad_to_max_length,
        )

        decoder_start_token = self._tokenizer.decode(decoder_start_token_id)
        self._decoder_assembler = Assembler(
            tokenizer=self._tokenizer,
            max_length=tgt_max_length,
            prefix_token_ids=decoder_start_token,
            pad_to_max_length=pad_to_max_length,
            add_special_tokens=False,
            return_token_type_ids=None,
        )
        self._label_assembler = Assembler(
            tokenizer=self._tokenizer,
            max_length=tgt_max_length,
            suffix_token_ids=self._tokenizer.eos_token,
            pad_to_max_length=pad_to_max_length,
            add_special_tokens=False,
            return_token_type_ids=None,
        )

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        key = self._indexes[index]
        item = self._data[key]
        splits = item.split("\t")
        src_outputs = self._src_assembler.batch_assemble([splits[0]])
        outputs = {
            "qids": torch.tensor([key]),
            "sum_input_ids": src_outputs["input_ids"][0],
            "sum_attention_mask": src_outputs["attention_mask"][0],
        }

        if len(splits) == 2:
            decoder_outputs = self._decoder_assembler.batch_assemble([splits[1]])
            label_outputs = self._label_assembler.batch_assemble([splits[1]])
            decoder_outputs = {
                "sum_decoder_input_ids": decoder_outputs["input_ids"][0],
                "sum_decoder_attention_mask": decoder_outputs["attention_mask"][0],
                "lm_labels": label_outputs["input_ids"][0],
            }
            outputs.update(decoder_outputs)

        assert all(x.dim() == 1 for x in outputs.values())
        return outputs
