from __future__ import annotations

from itertools import zip_longest
from typing import Dict, Generic, List, Optional, Sequence, TypeVar

import torch
from irtools.pad import pad_batch
from transformers import PreTrainedTokenizer

T = TypeVar("T", str, List[int])


class Assembler(Generic[T]):
    """Assemble pre-tokenized ids into shapes accepted by transformer models. This assembler
    should only be used for fine-tuning as it disregards different tokenizers'
    assembling logic and enforce a unified assembling approach. Using it for pre-trained
    model inference would result in undefined behavior.

    Single: [bos] + [prefix] + ids + [eos] + [optional pad]

    Paired: [bos] + [prefix] + first_ids + [sep] + second_ids [eos] + [optional pad]

    The `bos`, `sep`, `eos`, and `pad` tokens are self-explained and must be single
    integer.

    The `prefix` is used as task identifier and can be a list of integers. The same
    model may be used for ranking, summarization, or other tasks.

    """

    def __init__(
        self,
        *,  # Enforcing keyword arguments for clarity
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        prefix_token_ids: Optional[T] = None,
        suffix_token_ids: Optional[T] = None,
        pad_to_max_length: bool = True,
        add_special_tokens: bool = True,
        return_special_tokens_mask: bool = False,
        return_token_type_ids: Optional[bool] = True,
    ):
        if pad_to_max_length:
            assert isinstance(max_length, int)

        self._tokenizer = tokenizer
        self._max_length = max_length

        self._prefix: Optional[T] = prefix_token_ids
        if isinstance(self._prefix, str):
            self._prefix = self._prefix.strip()
            if not self._prefix.endswith(">"):
                self._prefix += " "

        self._suffix: Optional[T] = suffix_token_ids
        if isinstance(self._suffix, str) and not self._suffix.startswith(" "):
            self._suffix = self._suffix.strip()
            if not self._suffix.startswith("<"):
                self._suffix = " " + self._suffix

        self._pad_to_max_length = pad_to_max_length
        self._add_special_tokens = add_special_tokens
        self._return_special_tokens_mask = return_special_tokens_mask
        self._return_token_type_ids = return_token_type_ids
        self._pad_values = {
            "input_ids": self._tokenizer.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": self._tokenizer.pad_token_type_id,
            "special_tokens_mask": 1,
        }

    def batch_assemble(
        self,
        first_ids_seq: Sequence[T],
        second_ids_seq: Optional[Sequence[T]] = None,
    ) -> Dict[str, torch.Tensor]:
        if second_ids_seq is None:
            if self._prefix is not None:
                first_ids_seq = [self._prefix + x for x in first_ids_seq]
            if self._suffix is not None:
                first_ids_seq = [x + self._suffix for x in first_ids_seq]
            second_ids_seq = []
        else:
            if self._prefix is not None:
                first_ids_seq = [self._prefix + x for x in first_ids_seq]
            if self._suffix is not None:
                second_ids_seq = [x + self._suffix for x in second_ids_seq]

        outputs: List[Dict[str, torch.Tensor]] = []
        for first_ids, second_ids in zip_longest(first_ids_seq, second_ids_seq):
            output = self._tokenizer.encode_plus(
                first_ids,
                second_ids,
                add_special_tokens=self._add_special_tokens,
                max_length=self._max_length,
                pad_to_max_length=self._pad_to_max_length,
                return_tensors="pt",
                return_token_type_ids=self._return_token_type_ids,
                return_attention_mask=True,
                truncation="longest_first",
                return_special_tokens_mask=self._return_special_tokens_mask,
            )
            assert all(v.size(0) == 1 for v in output.values())
            outputs.append({k: v[0] for k, v in output.items()})
        encoded_output: Dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            padded = pad_batch(
                [one[key] for one in outputs], value=self._pad_values[key]
            )
            tensor = torch.stack(padded)
            encoded_output[key] = tensor

        return encoded_output
