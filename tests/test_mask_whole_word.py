from typing import Set

import pytest
from transformers import AutoTokenizer

from gdmtl.datasets import mask_difference, mask_whole_word


@pytest.mark.parametrize(  # type:ignore
    """arch,seq,prob,expected""",
    [
        ("bert-base-uncased", "binsheng", 1.0, {"[MASK] [MASK] [MASK]"}),
        (
            "bert-base-uncased",
            "binsheng binsheng",
            0.5,
            {"binsheng [MASK] [MASK] [MASK]", "[MASK] [MASK] [MASK] binsheng"},
        ),
    ],
)
def test_batch_assemble(arch: str, seq: str, prob: float, expected: Set[str]) -> None:
    tokenizer = AutoTokenizer.from_pretrained(arch)
    out = mask_whole_word(tokenizer, seq, prob)
    assert out in expected


@pytest.mark.parametrize(  # type:ignore
    """arch,seq,seq2,ratio,expected""",
    [
        (
            "bert-base-uncased",
            "hello binsheng",
            "binsheng",
            1.0,
            {"hello [MASK] [MASK] [MASK]"},
        ),
        (
            "bert-base-uncased",
            "hello binsheng",
            "hello binsheng",
            0.5,
            {"hello [MASK] [MASK] [MASK]", "[MASK] binsheng"},
        ),
    ],
)
def test_mask_difference(
    arch: str, seq: str, seq2: str, ratio: float, expected: Set[str]
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(arch)
    out = mask_difference(tokenizer, seq, seq2, ratio)
    assert out in expected
