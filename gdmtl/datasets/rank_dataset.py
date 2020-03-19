from __future__ import annotations

import logging
import multiprocessing as mp
import os
from abc import abstractmethod
from itertools import repeat
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from numpy_indexed import group_by
from scipy.special import softmax
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from gdmtl.utils import local_rank

from .assembler import Assembler
from .data_typing import DatasetT
from .qa_dataset import QADataset
from .tsv_dataset import TsvCollection
from .utils import load_run_file

log = logging.getLogger(__name__)


class TokenCountDataset(DatasetT):
    @abstractmethod
    def estimate_tokens(self, index: int) -> int:
        ...


class RankGroupDataset(TokenCountDataset):
    """Grouped documents for neural ranker training.

    - bos: tokenizer.bos_token_id
    - prefix: used to differentiate tasks. For example: tok.encode('rank:') or
              tok.encode('summary:')
    - sep: tokenizer.sep_token_id
    - eos: tokenizer.eos_token_id
    """

    @classmethod
    def create(
        cls,
        data: str,
        num_dup: int,
        num_neg: int,
        sample: Optional[Union[float, int]] = None,
        epochs: int = 1,
        qid_inc: int = 0,
        variants_ratio: float = 0.0,
        variants_start_epoch: int = 0,
        top10_prob: float = 0.01,
        **kwargs: Any,
    ) -> List[RankGroupDataset]:
        array = dict(np.load(data))

        # This is for debugging purpose. Quickly run past data preparation.
        if "NUM_TRAIN_SIZE" in os.environ:
            array = {
                k: v[: int(os.environ["NUM_TRAIN_SIZE"])] for k, v in array.items()
            }

        indexes = RankGroupDataset._make_groups(array, num_dup * epochs, num_neg)

        indexes = indexes.reshape(epochs, -1, (num_neg + 1))
        num_examples = indexes.shape[1]
        assert indexes.shape == (epochs, num_examples, (num_neg + 1))

        assert variants_ratio >= 0.0 and variants_ratio <= 1.0
        num_variants = int(num_examples * variants_ratio)
        # https://stackoverflow.com/a/45438143/955952
        qv_idx = np.random.rand(epochs - variants_start_epoch, num_examples).argsort(
            axis=1
        )[:, :num_variants]
        epc_indexes = np.broadcast_to(  # noqa
            np.arange(variants_start_epoch, epochs)[:, None], qv_idx.shape
        )

        assert indexes.shape == (epochs, num_examples, (num_neg + 1))

        if "qa_data" in kwargs and kwargs["qa_data"] is not None:
            kwargs["qa_data"] = QADataset(
                path=kwargs["qa_data"],
                tokenizer=kwargs["tokenizer"],
                max_length=kwargs["max_length"],
                prefix=kwargs.pop("qa_prefix"),
            )

        return [
            cls(
                array={k: v[x] for k, v in array.items()},
                num_dup=num_dup,
                num_neg=num_neg,
                sample=sample,
                **kwargs,
            )
            for x in indexes
        ]

    def __init__(
        self,
        array: Mapping[str, np.ndarray],
        tokenizer: PreTrainedTokenizer,
        query_col: TsvCollection,
        doc_col: TsvCollection,
        num_dup: int,
        num_neg: int,
        sample: Optional[Union[float, int]] = None,
        sort: Optional[str] = None,
        max_length: Optional[int] = None,
        prefix_token_ids: Optional[str] = None,
        pad_to_max_length: bool = True,
    ):
        assert sort in [None, "ascending", "descending"]
        assert array["qid"].shape[1:] == (num_neg + 1,)
        self._array = array
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._query_col = query_col
        self._doc_col = doc_col
        self._max_length = max_length
        self._assembler = Assembler(
            tokenizer=tokenizer,
            max_length=max_length,
            prefix_token_ids=prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )

        assert sort is None, "Sort training data not supported yet."

    def __getitem__(self, index: int) -> Dict[str, Any]:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        label = self._array["label"][index]
        assert qid.shape == (self._num_neg + 1,)

        outputs: Dict[str, Any] = self._assembler.batch_assemble(
            [self._query_col[x] for x in qid],
            [self._doc_col[x] for x in did],
        )
        outputs["qids"] = torch.tensor([int(x) for x in qid])
        outputs["dnos"] = torch.tensor([int(x) for x in did])
        outputs["labels"] = torch.tensor(label)
        return outputs

    def __len__(self) -> int:
        return len(self._array["qid"])

    @staticmethod
    def _make_groups(
        array: Mapping[str, np.ndarray], num_epochs: int, num_neg: int
    ) -> np.ndarray:
        indexes = group_by(array["qid"], np.arange(len(array["qid"])))[1]

        # Randomness stays out of multiprocessing for reproducibility.
        with mp.Pool(mp.cpu_count() // 2) as pool:
            mapping = pool.imap(
                RankGroupDataset._make_one_group,
                zip(indexes, (array["label"][x] for x in indexes), repeat(1000)),
                chunksize=1024,
            )
            result = list(
                tqdm(
                    mapping,
                    total=len(indexes),
                    desc=f"{local_rank()}: Sampling",
                    position=local_rank(),
                )
            )

        pos_idx = np.concatenate([x[0] for x in result])
        total_pos = len(pos_idx)

        neg_idx = np.concatenate([x[1] for x in result])
        assert neg_idx.shape == (total_pos, 1000)

        neg_weights = np.concatenate([x[2] for x in result])
        assert neg_weights.shape == (total_pos, 1000)

        neg_weights = torch.as_tensor(neg_weights)
        index_tensor = torch.multinomial(neg_weights, num_epochs * num_neg, True)
        assert index_tensor.shape == (total_pos, num_epochs * num_neg)

        dim0 = np.broadcast_to(np.arange(len(neg_idx))[:, None], index_tensor.shape)

        neg_idx = neg_idx[dim0, index_tensor]
        assert neg_idx.shape == (total_pos, num_epochs * num_neg)

        pos_idx = np.repeat(pos_idx[:, None, None], repeats=num_epochs, axis=1)
        assert pos_idx.shape == (total_pos, num_epochs, 1)

        neg_idx = neg_idx.reshape(total_pos, num_epochs, num_neg)
        assert neg_idx.shape == (total_pos, num_epochs, num_neg)

        groups = np.concatenate((pos_idx, neg_idx), axis=2)
        assert groups.shape == (total_pos, num_epochs, num_neg + 1)

        log.debug("Logging data sample to training-data.tsv")
        with open("training-data.tsv", "w") as f:
            for group_idx in groups.reshape(-1, num_neg + 1):
                group_pos_idx = group_idx[0]
                group_qid = array["qid"][group_pos_idx]
                for group_neg_idx in group_idx[1:]:
                    group_pos_did = array["did"][group_pos_idx]
                    group_neg_did = array["did"][group_neg_idx]
                    line = f"{group_qid}\t{group_pos_did}\t{group_neg_did}\n"
                    f.write(line)

        groups = np.transpose(groups, (1, 0, 2))
        assert groups.shape == (num_epochs, total_pos, num_neg + 1)

        return groups

    @staticmethod
    def _make_one_group(zipped: Sequence[Any]) -> np.ndarray:
        index_arr, label_arr, num_neg_align = zipped
        labels = sorted(np.unique(label_arr), reverse=True)

        all_pos_index = np.empty((0,), dtype=int)
        all_neg_index = np.empty((0, num_neg_align), dtype=int)
        all_weights = np.empty((0, num_neg_align), dtype=float)

        for label in labels[:-1]:
            pos_index = index_arr[label_arr == label]
            all_pos_index = np.concatenate((all_pos_index, pos_index))

            neg_index = index_arr[label_arr < label]

            weights = np.full((len(neg_index),), 1.0 / len(neg_index))

            if len(neg_index) < num_neg_align:
                pad = num_neg_align - len(neg_index)
                neg_index = np.pad(neg_index, (0, pad), constant_values=0)
                weights = np.pad(weights, (0, pad), constant_values=0)
            else:
                neg_index = neg_index[:num_neg_align]
                weights = weights[:num_neg_align]

            neg_index = np.broadcast_to(
                neg_index[None, ...], (len(pos_index), *neg_index.shape)
            )
            weights = np.broadcast_to(
                weights[None, ...], (len(pos_index), *weights.shape)
            )

            all_neg_index = np.concatenate((all_neg_index, neg_index))
            all_weights = np.concatenate((all_weights, weights))

        total_pos = len(all_pos_index)

        assert all_pos_index.shape == (total_pos,)
        assert all_neg_index.shape == (total_pos, num_neg_align)
        assert all_weights.shape == (total_pos, num_neg_align)

        return all_pos_index, all_neg_index, all_weights

    @staticmethod
    def _make_one_group_softmax(zipped: Sequence[Any]) -> np.ndarray:
        current, num_neg, top10_prob = zipped
        labels = sorted(np.unique(current[:, -1]), reverse=True)
        assert current.shape[1] == 4
        if len(labels) == 1:
            return (
                np.empty((0, 3), dtype=int),
                np.empty((0, num_neg, 3), dtype=int),
                np.empty((0, num_neg), dtype=float),
            )

        temp = 1.0
        all_pos = np.empty((0, 3), dtype=int)
        all_neg = np.empty((0, num_neg, 3), dtype=int)
        all_probs = np.empty((0, num_neg), dtype=float)
        for label in labels[:-1]:
            pos = current[current[:, -1] == label][:, [0, 1, 3]].astype(int)

            negs = current[current[:, -1] < label]
            logits = negs[:, 2]
            negs = negs[:, [0, 1, 3]].astype(int)
            probs = softmax(logits / temp)
            loc = 10
            if len(probs) > 100 and probs[:loc].sum() > top10_prob:
                while probs[:loc].sum() > top10_prob:
                    temp += 0.1
                    probs = softmax(logits / temp)
            elif len(probs) > 100 and probs[:loc].sum() < top10_prob:
                while probs[:loc].sum() < top10_prob:
                    temp -= 0.1
                    probs = softmax(logits / temp)
            if len(negs) < num_neg:
                pad = num_neg - len(negs)
                negs = np.pad(negs, ((0, pad), (0, 0)), constant_values=0)
                probs = np.pad(probs, (0, pad), constant_values=0)
            negs = np.broadcast_to(negs[None, ...], (len(pos), *negs.shape))
            probs = np.broadcast_to(probs[None, ...], (len(pos), *probs.shape))

            all_pos = np.vstack((all_pos, pos))
            all_neg = np.vstack((all_neg, negs))
            all_probs = np.vstack((all_probs, probs))

        total_pos = len(all_pos)
        assert all_pos.shape == (total_pos, 3)
        assert all_neg.shape == (total_pos, num_neg, 3)
        assert all_probs.shape == (total_pos, num_neg)
        return all_pos, all_neg, all_probs

    def estimate_tokens(self, index: int) -> int:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        assert qid.shape == (self._num_neg + 1,)
        lengths = np.array(
            [
                8 + len(self._query_col[x]) + len(self._doc_col[y])
                for x, y in zip(qid, did)
            ]
        )
        size: int = min(lengths.max(), self._max_length) * (self._num_neg + 1)
        return size


class RankPointDataset(TokenCountDataset):
    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer,
        query_col: TsvCollection,
        doc_col: TsvCollection,
        src_max_length: int,
        pad_to_max_length: bool = True,
        sample: Optional[Union[float, int]] = None,
        sort: Optional[str] = None,
        rank_prefix_token_ids: Optional[str] = None,
        first_seq: str = "query",
        **kwargs: Any,
    ):
        assert sort in [None, "ascending", "descending"]
        assert first_seq in ["query", "passage"]
        if kwargs:
            log.warning(f"Unused params {kwargs}")

        if isinstance(data, str):
            if data.endswith(".npy"):
                array = np.load(data)
                self._array = {
                    "qid": array[:, 0],
                    "did": array[:, 1],
                    "label": array[:, 2],
                }
            elif data.endswith(".npz"):
                self._array = dict(np.load(data))
            elif data.endswith(".ans_run") or data.endswith(".run"):
                self._array = load_run_file(data)
        else:
            raise ValueError("Unrecognized data type")
        # This is for debugging purpose. Quickly run past data preparation.
        if "NUM_VALID_SIZE" in os.environ:
            self._array = {
                k: v[: int(os.environ["NUM_VALID_SIZE"])]
                for k, v in self._array.items()
            }

        self._first_seq = first_seq
        self._query_col = query_col
        self._doc_col = doc_col
        self._max_length = src_max_length
        self._assembler = Assembler(
            tokenizer=tokenizer,
            max_length=src_max_length,
            prefix_token_ids=rank_prefix_token_ids,
            pad_to_max_length=pad_to_max_length,
        )

        if sort is not None:
            log.info("Sort documents by length")
            lengths = np.zeros(len(self._array["qid"]))
            for i, (qid, did) in enumerate(zip(self._array["qid"], self._array["did"])):
                lengths[i] = self._query_col.tokens(qid) + self._doc_col.tokens(did)

            sorted_indexes = np.argsort(lengths)
            if sort == "descending":
                sorted_indexes = sorted_indexes[::-1]
            self._array = {k: v[sorted_indexes] for k, v in self._array.items()}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        label = self._array["label"][index]
        if self._first_seq == "query":
            first_ids_seq = [self._query_col[qid]]
            second_ids_seq = [self._doc_col[did]]
        elif self._first_seq == "passage":
            first_ids_seq = [self._doc_col[did]]
            second_ids_seq = [self._query_col[qid]]
        else:
            assert False, f"Logic bug first seq: {self._first_seq}"

        outputs: Dict[str, Any] = self._assembler.batch_assemble(
            first_ids_seq, second_ids_seq
        )
        outputs = {
            "qids": torch.tensor([int(qid)]),
            "dnos": torch.tensor([int(did)]),
            "labels": torch.tensor([label]),
            "rank_input_ids": outputs["input_ids"][0],
            "rank_attention_mask": outputs["attention_mask"][0],
            "rank_token_type_ids": outputs["token_type_ids"][0],
        }

        assert outputs["labels"].dim() == 1
        assert outputs["rank_input_ids"].dim() == 1
        assert outputs["rank_attention_mask"].dim() == 1
        assert outputs["rank_token_type_ids"].dim() == 1

        return outputs

    def __len__(self) -> int:
        return len(self._array["qid"])

    def estimate_tokens(self, index: int) -> int:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        size: int = 8 + len(self._query_col[qid]) + len(self._doc_col[did])
        return min(size, self._max_length)


class RankPointSepDataset(RankPointDataset):
    def __init__(
        self,
        data: str,
        tokenizer: PreTrainedTokenizer,
        query_col: TsvCollection,
        doc_col: TsvCollection,
        src_max_length: int,
        tgt_max_length: int,
        decoder_start_token_id: int,
        pad_to_max_length: bool = True,
        sample: Optional[Union[float, int]] = None,
        sort: Optional[str] = None,
        rank_prefix_token_ids: Optional[str] = None,
        first_seq: str = "query",
        **kwargs: Any,
    ):
        if kwargs:
            log.warning(f"Unused params {kwargs}")
        super().__init__(
            data,
            tokenizer,
            query_col,
            doc_col,
            src_max_length,
            pad_to_max_length,
            sample,
            sort,
            rank_prefix_token_ids,
            first_seq,
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

    def __getitem__(self, index: int) -> Dict[str, Any]:
        qid = self._array["qid"][index]
        did = self._array["did"][index]
        label = self._array["label"][index]

        queries = [self._query_col[qid]]
        passages = [self._doc_col[did]]

        rank_inputs = self._assembler.batch_assemble(passages)
        rank_decoder_inputs = self._decoder_assembler.batch_assemble(queries)

        item: Dict[str, Any] = {
            "qids": torch.tensor([int(qid)]),
            "dnos": torch.tensor([int(did)]),
            "rank_input_ids": rank_inputs["input_ids"][0],
            "rank_attention_mask": rank_inputs["attention_mask"][0],
            "rank_decoder_input_ids": rank_decoder_inputs["input_ids"][0],
            "rank_decoder_attention_mask": rank_decoder_inputs["attention_mask"][0],
            "labels": torch.tensor([label]),
        }

        assert item["rank_input_ids"].dim() == 1
        assert item["rank_attention_mask"].dim() == 1
        assert item["rank_decoder_input_ids"].dim() == 1
        assert item["rank_decoder_attention_mask"].dim() == 1
        assert item["labels"].dim() == 1

        return item
