from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from .assembler import Assembler
from .qa_dataset import QADataset
from .rank_dataset import RankGroupDataset
from .tsv_dataset import TsvCollection
from .utils import (
    make_targets_mlm_inputs,
    make_targets_ntp_inputs,
    mask_difference,
    mask_whole_word,
)

log = logging.getLogger(__name__)


class MtlSepDataset(RankGroupDataset):
    def __init__(
        self,
        array: Mapping[str, np.ndarray],
        tokenizer: PreTrainedTokenizer,
        query_col: TsvCollection,
        doc_col: TsvCollection,
        num_dup: int,
        num_neg: int,
        decoder_start_token_id: int,
        src_max_length: int,
        tgt_max_length: int,
        sample: Optional[Union[float, int]] = None,
        sort: Optional[str] = None,
        max_length: Optional[int] = None,
        summarizer_prefix_token_ids: Optional[str] = None,
        rank_prefix_token_ids: Optional[str] = None,
        pad_to_max_length: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            log.warning(f"Unused parameters: {kwargs}")
        super().__init__(
            array,
            tokenizer,
            query_col,
            doc_col,
            num_dup,
            num_neg,
            sample,
            sort,
            max_length,
            summarizer_prefix_token_ids,
            pad_to_max_length,
        )
        self._pas_pad = tokenizer.pad_token_id
        self._sum_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=summarizer_prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )
        self._rank_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=rank_prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )

        decoder_start_token = tokenizer.decode(decoder_start_token_id)
        self._decoder_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=tgt_max_length,
            prefix_token_ids=decoder_start_token,
            pad_to_max_length=False,
            add_special_tokens=False,
            return_token_type_ids=None,
        )
        self._label_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=tgt_max_length,
            suffix_token_ids=tokenizer.eos_token,
            pad_to_max_length=False,
            add_special_tokens=False,
            return_token_type_ids=None,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        label = self._array["label"][index]
        assert qid.shape == (self._num_neg + 1,)
        assert did.shape == (self._num_neg + 1,)
        assert label.shape == (self._num_neg + 1,)

        queries = [self._query_col[x] for x in qid]
        passages = [self._doc_col[x] for x in did]

        sum_inputs = self._sum_assembler.batch_assemble(passages)
        sum_decoder_inputs = self._decoder_assembler.batch_assemble(queries)
        lm_labels = self._label_assembler.batch_assemble(queries)
        lm_labels["input_ids"].masked_fill_(~lm_labels["attention_mask"].bool(), -100)

        rank_inputs = self._rank_assembler.batch_assemble(passages)
        rank_decoder_inputs = self._decoder_assembler.batch_assemble(queries)

        item: Dict[str, Any] = {
            "qids": torch.tensor([int(x) for x in qid]),
            "dnos": torch.tensor([int(x) for x in did]),
            "sum_input_ids": sum_inputs["input_ids"],
            "sum_attention_mask": sum_inputs["attention_mask"],
            "sum_decoder_input_ids": sum_decoder_inputs["input_ids"],
            "sum_decoder_attention_mask": sum_decoder_inputs["attention_mask"],
            "rank_input_ids": rank_inputs["input_ids"],
            "rank_attention_mask": rank_inputs["attention_mask"],
            "rank_decoder_input_ids": rank_decoder_inputs["input_ids"],
            "rank_decoder_attention_mask": rank_decoder_inputs["attention_mask"],
            "lm_labels": lm_labels["input_ids"],
        }

        assert item["sum_input_ids"].dim() == 2
        assert item["sum_attention_mask"].dim() == 2
        assert item["sum_decoder_input_ids"].dim() == 2
        assert item["sum_decoder_attention_mask"].dim() == 2
        assert item["rank_input_ids"].dim() == 2
        assert item["rank_attention_mask"].dim() == 2
        assert item["rank_decoder_input_ids"].dim() == 2
        assert item["rank_decoder_attention_mask"].dim() == 2
        assert item["lm_labels"].dim() == 2

        return item


class MtlMixedDataset(RankGroupDataset):
    def __init__(
        self,
        array: Mapping[str, np.ndarray],
        tokenizer: PreTrainedTokenizer,
        query_col: TsvCollection,
        doc_col: TsvCollection,
        num_dup: int,
        num_neg: int,
        src_max_length: int,
        sample: Optional[Union[float, int]] = None,
        sort: Optional[str] = None,
        max_length: Optional[int] = None,
        summarizer_prefix_token_ids: Optional[str] = None,
        rank_prefix_token_ids: Optional[str] = None,
        pad_to_max_length: bool = True,
        qa_data: Optional[Union[QADataset, str]] = None,
        qa_prefix: str = "",
        mask_whole_word_prob: float = 0.0,
        mask_qgen_query: bool = False,
        mask_query_from_passage: float = 0.0,
        min_rel_for_qgen: int = 1,
        **kwargs: Any,
    ):
        if kwargs:
            log.warning(f"Unused params {kwargs}")
        super(MtlMixedDataset, self).__init__(
            array,
            tokenizer,
            query_col,
            doc_col,
            num_dup,
            num_neg,
            sample,
            sort,
            max_length,
            summarizer_prefix_token_ids,
            pad_to_max_length,
        )

        self._pas_pad = tokenizer.pad_token_id
        self._tokenizer = tokenizer
        self._mask_whole_word_prob = mask_whole_word_prob
        self._mask_query_from_passage = mask_query_from_passage
        self._mask_qgen_query = mask_qgen_query
        self._min_rel_for_qgen = min_rel_for_qgen
        self._sum_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=summarizer_prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )
        self._rank_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=rank_prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )

        if qa_data is not None:
            if isinstance(qa_data, str):
                self._qa = QADataset(
                    path=qa_data,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    prefix=qa_prefix,
                )
            else:
                self._qa = qa_data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        label = self._array["label"][index]
        assert qid.shape == (self._num_neg + 1,)
        assert did.shape == (self._num_neg + 1,)
        assert label.shape == (self._num_neg + 1,)

        if label[0] < self._min_rel_for_qgen:
            idx = (self._array["label"][:, 0] >= self._min_rel_for_qgen).nonzero()[0]
            sample = np.random.choice(idx)
            qgen_queries = [self._query_col[x] for x in self._array["qid"][sample]]
            passages = [self._doc_col[x] for x in self._array["did"][sample]]
            qgen_passages = [self._doc_col[x] for x in self._array["did"][sample]]
            sum_input_weights = torch.tensor(
                self._array["label"][sample][:1], dtype=torch.float
            )
        else:
            qgen_queries = [self._query_col[x] for x in qid]
            passages = [self._doc_col[x] for x in did]
            qgen_passages = [self._doc_col[x] for x in did]
            sum_input_weights = torch.tensor(label[:1], dtype=torch.float)

        if self._mask_query_from_passage > 0.0:
            qgen_passages = [
                mask_difference(self._tokenizer, x, y, self._mask_query_from_passage)
                for x, y in zip(qgen_passages, qgen_queries)
            ]

        if self._mask_whole_word_prob > 0:
            qgen_passages = [
                mask_whole_word(self._tokenizer, x, self._mask_whole_word_prob)
                for x in qgen_passages
            ]

        if self._mask_qgen_query:
            sum_inputs = make_targets_mlm_inputs(
                self._assembler,
                self._tokenizer,
                passages[:1],
                qgen_queries[:1],
                qgen_passages[:1],
            )
        else:
            sum_inputs = make_targets_ntp_inputs(
                self._assembler,
                self._tokenizer,
                passages[:1],
                qgen_queries[:1],
                qgen_passages[:1],
            )

        rank_queries = [self._query_col[x] for x in qid]
        rank_passages = [self._doc_col[x] for x in did]
        rank_inputs = self._rank_assembler.batch_assemble(rank_passages, rank_queries)

        item: Dict[str, Any] = {
            "qids": torch.tensor([int(x) for x in qid]),
            "dnos": torch.tensor([int(x) for x in did]),
            "sum_input_ids": sum_inputs["input_ids"],
            "sum_token_type_ids": sum_inputs["token_type_ids"],
            "sum_attention_mask": sum_inputs["attention_mask"],
            "sum_input_weights": sum_input_weights,
            "rank_input_ids": rank_inputs["input_ids"],
            "rank_token_type_ids": rank_inputs["token_type_ids"],
            "rank_attention_mask": rank_inputs["attention_mask"],
            "lm_labels": sum_inputs["lm_labels"],
        }

        assert item["sum_input_ids"].dim() == 2
        if self._mask_qgen_query:
            assert item["sum_attention_mask"].dim() == 2
        else:
            assert item["sum_attention_mask"].dim() == 3
        assert item["sum_token_type_ids"].dim() == 2
        assert item["sum_input_weights"].dim() == 1

        assert item["rank_input_ids"].dim() == 2
        assert item["rank_token_type_ids"].dim() == 2
        assert item["rank_attention_mask"].dim() == 2
        assert item["lm_labels"].dim() == 2

        if hasattr(self, "_qa"):
            pos_qid = qid[0]
            qa_inputs = {f"qa_{k}": v for k, v in self._qa.by_qid(pos_qid).items()}
            item.update(qa_inputs)

        return item


class MtlCatDataset(RankGroupDataset):
    def __init__(
        self,
        array: Mapping[str, np.ndarray],
        tokenizer: PreTrainedTokenizer,
        query_col: TsvCollection,
        doc_col: TsvCollection,
        num_dup: int,
        num_neg: int,
        src_max_length: int,
        tgt_max_length: int,
        decoder_start_token_id: int,
        sample: Optional[Union[float, int]] = None,
        sort: Optional[str] = None,
        max_length: Optional[int] = None,
        summarizer_prefix_token_ids: Optional[str] = None,
        rank_prefix_token_ids: Optional[str] = None,
        pad_to_max_length: bool = True,
        **kwargs: Any,
    ):
        if kwargs:
            log.warning(f"Unused params {kwargs}")
        super().__init__(
            array,
            tokenizer,
            query_col,
            doc_col,
            num_dup,
            num_neg,
            sample,
            sort,
            max_length,
            summarizer_prefix_token_ids,
            pad_to_max_length,
        )

        self._pas_pad = tokenizer.pad_token_id
        self._tokenizer = tokenizer
        self._sum_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=summarizer_prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )
        self._rank_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=rank_prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )

        decoder_start_token = tokenizer.decode(decoder_start_token_id)
        self._decoder_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=tgt_max_length,
            prefix_token_ids=decoder_start_token,
            pad_to_max_length=False,
            add_special_tokens=False,
            return_token_type_ids=None,
        )
        self._label_assembler = Assembler(
            tokenizer=tokenizer,
            max_length=tgt_max_length,
            suffix_token_ids=tokenizer.eos_token,
            pad_to_max_length=False,
            add_special_tokens=False,
            return_token_type_ids=None,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        label = self._array["label"][index]
        assert qid.shape == (self._num_neg + 1,)
        assert did.shape == (self._num_neg + 1,)
        assert label.shape == (self._num_neg + 1,)

        queries = [self._query_col[x] for x in qid]
        passages = [self._doc_col[x] for x in did]

        sum_inputs = self._sum_assembler.batch_assemble(passages)
        sum_decoder_inputs = self._decoder_assembler.batch_assemble(queries)
        lm_labels = self._label_assembler.batch_assemble(queries)
        lm_labels["input_ids"].masked_fill_(~lm_labels["attention_mask"].bool(), -100)

        rank_passages = [self._doc_col[x] for x in did]
        rank_inputs = self._rank_assembler.batch_assemble(rank_passages, queries)

        item: Dict[str, Any] = {
            "qids": torch.tensor([int(x) for x in qid]),
            "dnos": torch.tensor([int(x) for x in did]),
            "sum_input_ids": sum_inputs["input_ids"],
            "sum_attention_mask": sum_inputs["attention_mask"],
            "sum_decoder_input_ids": sum_decoder_inputs["input_ids"],
            "sum_decoder_attention_mask": sum_decoder_inputs["attention_mask"],
            "rank_input_ids": rank_inputs["input_ids"],
            "rank_attention_mask": rank_inputs["attention_mask"],
            "lm_labels": lm_labels["input_ids"],
        }

        assert item["sum_input_ids"].dim() == 2
        assert item["sum_attention_mask"].dim() == 2
        assert item["sum_decoder_input_ids"].dim() == 2
        assert item["sum_decoder_attention_mask"].dim() == 2

        assert item["rank_input_ids"].dim() == 2
        assert item["rank_attention_mask"].dim() == 2
        assert item["lm_labels"].dim() == 2

        return item
