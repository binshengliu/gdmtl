from typing import Optional, Sequence

import pytest
import torch
from transformers import AutoTokenizer

from gdmtl.datasets import Assembler, make_targets_mlm_inputs, make_targets_ntp_inputs

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
assembler = Assembler(
    tokenizer=tokenizer,
    max_length=50,
    prefix_token_ids=None,
    pad_to_max_length=False,
)


@pytest.mark.parametrize(  # type:ignore
    """src,src_with_mask,tgt,inputs,labels,attention_mask""",
    [
        (
            "hello this is the source",
            None,
            "target to predict",
            "[CLS] hello this is the source [SEP] [MASK] [MASK] [MASK] [SEP]",
            "[UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] target to predict [SEP]",
            torch.tensor([1] * 11),
        ),
        (
            "hello this is the source",
            "hello [MASK] is the source",
            "target to predict",
            "[CLS] hello [MASK] is the source [SEP] [MASK] [MASK] [MASK] [SEP]",
            "[UNK] [UNK] this [UNK] [UNK] [UNK] [UNK] target to predict [SEP]",
            torch.tensor([1] * 11),
        ),
    ],
)
def test_mlm(
    src: str,
    src_with_mask: Optional[str],
    tgt: str,
    inputs: str,
    labels: str,
    attention_mask: torch.Tensor,
) -> None:
    result = make_targets_mlm_inputs(
        assembler, tokenizer, [src], [tgt], [src_with_mask] if src_with_mask else None
    )
    assert tokenizer.decode(result["input_ids"][0]) == inputs
    assert tokenizer.decode(result["lm_labels"][0]) == labels
    assert torch.equal(result["attention_mask"][0], attention_mask)


@pytest.mark.parametrize(  # type:ignore
    """src,src_with_mask,tgt,inputs,labels,attention_mask""",
    [
        (
            "hello this is the source",
            None,
            "target to predict",
            "[CLS] hello this is the source [SEP] target to predict [SEP]",
            "[UNK] [UNK] [UNK] [UNK] [UNK] [UNK] target to predict [SEP] [UNK]",
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        ),
        (
            "hello this is the source",
            "hello this is the [MASK]",
            "target to predict",
            "[CLS] hello this is the [MASK] [SEP] target to predict [SEP]",
            "[UNK] [UNK] [UNK] [UNK] [UNK] source target to predict [SEP] [UNK]",
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        ),
    ],
)
def test_ntp(
    src: str,
    src_with_mask: Optional[str],
    tgt: str,
    inputs: str,
    labels: str,
    attention_mask: torch.Tensor,
) -> None:
    result = make_targets_ntp_inputs(
        assembler, tokenizer, [src], [tgt], [src_with_mask] if src_with_mask else None
    )
    assert tokenizer.decode(result["input_ids"][0]) == inputs
    assert tokenizer.decode(result["lm_labels"][0]) == labels
    assert torch.equal(result["attention_mask"][0], attention_mask)


@pytest.mark.parametrize(  # type:ignore
    """src,tgt,inputs,labels,attention_mask""",
    [
        (
            ["short one", "relatively longer one"],
            ["target", "target"],
            [
                "[CLS] short one [SEP] target [SEP] [PAD]",
                "[CLS] relatively longer one [SEP] target [SEP]",
            ],
            [
                "[UNK] [UNK] [UNK] target [SEP] [UNK] [UNK]",
                "[UNK] [UNK] [UNK] [UNK] target [SEP] [UNK]",
            ],
            [
                [
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1],
                ],
            ],
        ),
    ],
)
def test_ntp_batch(
    src: Sequence[str],
    tgt: Sequence[str],
    inputs: Sequence[str],
    labels: Sequence[str],
    attention_mask: torch.Tensor,
) -> None:
    result = make_targets_ntp_inputs(assembler, tokenizer, src, tgt, None)

    assert tokenizer.decode(result["input_ids"][0]) == inputs[0]
    assert tokenizer.decode(result["lm_labels"][0]) == labels[0]
    assert torch.equal(result["attention_mask"][0], torch.tensor(attention_mask[0]))

    assert tokenizer.decode(result["input_ids"][1]) == inputs[1]
    assert tokenizer.decode(result["lm_labels"][1]) == labels[1]
    assert torch.equal(result["attention_mask"][1], torch.tensor(attention_mask[1]))
