from typing import Dict, List, Optional, Sequence

import pytest
import torch
from transformers import AutoTokenizer

from gdmtl.datasets import Assembler


@pytest.mark.parametrize(  # type:ignore
    """arch,first_ids_seq,second_ids_seq,max_length,prefix_token_ids,suffix_token_ids,
    pad_to_max_length,add_special_tokens,outputs""",
    [
        (
            "bert-base-uncased",
            [[10, 20, 30], [10, 20, 30, 40]],
            [[40, 50, 60, 70], [40, 50, 60, 70]],
            15,
            [77, 88],
            [],
            False,
            True,
            {
                "input_ids": torch.tensor(
                    [
                        [101, 77, 88, 10, 20, 30, 102, 40, 50, 60, 70, 102, 0],
                        [101, 77, 88, 10, 20, 30, 40, 102, 40, 50, 60, 70, 102],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ]
                ),
                "token_type_ids": torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    ]
                ),
            },
        ),
        (
            "bert-base-uncased",
            [[10, 20, 30], [10, 20, 30, 40]],
            [[40, 50, 60, 70], [40, 50, 60, 70]],
            15,
            [77, 88],
            [],
            True,
            True,
            {
                "input_ids": torch.tensor(
                    [
                        [101, 77, 88, 10, 20, 30, 102, 40, 50, 60, 70, 102, 0, 0, 0],
                        [101, 77, 88, 10, 20, 30, 40, 102, 40, 50, 60, 70, 102, 0, 0],
                    ]
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    ]
                ),
                "token_type_ids": torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                    ]
                ),
            },
        ),
        (
            "bert-base-uncased",
            [[10, 20, 30], [10, 20, 30, 40]],
            None,
            15,
            [77, 88],
            [102],
            False,
            False,
            {
                "input_ids": torch.tensor(
                    [[77, 88, 10, 20, 30, 102, 0], [77, 88, 10, 20, 30, 40, 102]]
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]
                ),
                "token_type_ids": torch.tensor(
                    [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
                ),
            },
        ),
    ],
)
def test_batch_assemble(
    arch: str,
    first_ids_seq: Sequence[List[int]],
    second_ids_seq: Optional[Sequence[List[int]]],
    max_length: Optional[int],
    prefix_token_ids: List[int],
    suffix_token_ids: List[int],
    pad_to_max_length: bool,
    add_special_tokens: bool,
    outputs: Dict[str, torch.Tensor],
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(arch)
    ids = Assembler(
        tokenizer=tokenizer,
        max_length=max_length,
        prefix_token_ids=prefix_token_ids,
        suffix_token_ids=suffix_token_ids,
        pad_to_max_length=pad_to_max_length,
        add_special_tokens=add_special_tokens,
    ).batch_assemble(first_ids_seq, second_ids_seq)
    assert ids.keys() == outputs.keys()
    assert torch.equal(ids["input_ids"], outputs["input_ids"])
    assert torch.equal(ids["attention_mask"], outputs["attention_mask"])
    assert torch.equal(ids["token_type_ids"], outputs["token_type_ids"])
