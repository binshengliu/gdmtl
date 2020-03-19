import string
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from more_itertools import flatten
from transformers import PreTrainedTokenizer

from gdmtl.datasets import Assembler


def get_dynamic_gradient_accumulation(
    batch_sizes: Sequence[int], effective_bsz: int
) -> List[int]:
    current_effective = 0
    current_acc = 0
    grad_accs = []
    for bsz in batch_sizes:
        current_effective += bsz
        current_acc += 1
        if current_effective >= effective_bsz:
            grad_accs.extend([current_acc] + [0] * (current_acc - 1))
            current_acc = 0
            current_effective = 0
    if current_acc > 0:
        grad_accs.extend([current_acc] + [0] * (current_acc - 1))
        current_acc = 0
        current_effective = 0
    assert len(grad_accs) == len(batch_sizes)
    return grad_accs


def unpack(array: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    if array.ndim == 2:
        return len(array), [array.shape[1]] * len(array), array

    total = array[0]
    lens = array[1:][:total]
    data = array[total + 1 :]

    splits = np.cumsum(lens)[:-1]
    payload = np.split(data, splits)
    return total, lens, payload


def sample_by_key(
    arr: np.ndarray, sample: Union[float, int], key_field: int = 0
) -> np.ndarray:
    assert arr.ndim == 2
    assert key_field <= arr.shape[1]
    assert sample > 0

    keys = np.unique(arr[:, key_field])
    size = len(keys)
    new_size = int(size * sample) if sample < 1.0 else sample
    assert new_size < size

    idx = np.random.choice(size, new_size, replace=False)
    keys = set(keys[idx])
    mask = [x in keys for x in arr[:, key_field]]
    arr = arr[mask]
    return arr


def mix_attention_mask(
    attn_mask: torch.Tensor, token_types: torch.Tensor
) -> torch.Tensor:
    # Bidirectional mask
    self_attn_mask = attn_mask.bool() & (token_types == 0)
    self_attn_mask = self_attn_mask[..., :, None] & self_attn_mask[..., None, :]

    # Unidirectional mask
    causal_mask = torch.tril(attn_mask[..., :, None] & attn_mask[..., None, :]).long()

    # Overwrite parts of unidirectional mask with bidirectional mask
    mixed_mask = causal_mask.masked_fill(self_attn_mask, 1)
    return mixed_mask


def make_lm_labels(
    tok: PreTrainedTokenizer,
    inputs: Dict[str, torch.Tensor],
    masked_inputs: Optional[torch.Tensor] = None,
    shift: bool = True,
) -> torch.Tensor:
    assert inputs["token_type_ids"].unique().numel() > 1  # type: ignore

    # 1. do not predict source
    labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)

    # 2. Shift for causal language modeling
    if shift:
        labels = labels.roll(-1, 1)

    if masked_inputs is None:
        return labels

    # 3. Predict [MASK] in source if present
    mlm_mask = (masked_inputs == tok.mask_token_id) & (inputs["token_type_ids"] == 0)
    labels[mlm_mask] = inputs["input_ids"][mlm_mask]
    return labels


def split_sequence(
    tok: PreTrainedTokenizer, seq: str
) -> Tuple[List[str], List[List[str]]]:
    tokens = tok.tokenize(seq)

    word_begin_mask = [not x.startswith("##") for x in tokens]
    split_loc = np.nonzero(word_begin_mask)[0]
    sub_words = [x.tolist() for x in np.split(tokens, split_loc[1:])]
    whole_words = [tok.convert_tokens_to_string(x) for x in sub_words]
    return whole_words, sub_words


def mask_whole_word(tok: PreTrainedTokenizer, seq: str, prob: float) -> str:
    _, sub_words = split_sequence(tok, seq)

    num_masks = int(len(sub_words) * prob)
    mask_index = np.random.choice(len(sub_words), num_masks, replace=False)
    for idx in mask_index:
        sub_words[idx] = [tok.mask_token] * len(sub_words[idx])

    masked_seq: str = tok.convert_tokens_to_string(list(flatten(sub_words)))
    return masked_seq


def mask_difference(
    tok: PreTrainedTokenizer, seq1: str, seq2: str, ratio: float
) -> str:
    whole_words, sub_words = split_sequence(tok, seq1)

    import spacy.lang.en.stop_words as stop_words

    def keep(x: str) -> bool:
        return x not in string.punctuation and x not in stop_words.STOP_WORDS

    seq2_words = list(set(split_sequence(tok, seq2)[0]))
    seq2_words = [x for x in seq2_words if keep(x)]
    seq2_words = np.random.choice(seq2_words, int(len(seq2_words) * ratio)).tolist()
    for i in range(len(whole_words)):
        if whole_words[i] in seq2_words:
            sub_words[i] = [tok.mask_token] * len(sub_words[i])

    masked_seq: str = tok.convert_tokens_to_string(list(flatten(sub_words)))
    return masked_seq


def make_targets_mlm_inputs(
    assembler: Assembler,
    tok: PreTrainedTokenizer,
    src: Sequence[str],
    tgt: Sequence[str],
    src_with_mask: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    if src_with_mask is None:
        src_with_mask = src

    masked_tgt = [mask_whole_word(tok, x, 1.0) for x in tgt]
    inputs: Dict[str, torch.Tensor] = assembler.batch_assemble(
        src_with_mask, masked_tgt
    )
    labels = assembler.batch_assemble(src, tgt)
    inputs["lm_labels"] = make_lm_labels(tok, labels, inputs["input_ids"], shift=False)
    return inputs


def make_targets_ntp_inputs(
    assembler: Assembler,
    tok: PreTrainedTokenizer,
    src: Sequence[str],
    tgt: Sequence[str],
    src_with_mask: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    if src_with_mask is None:
        src_with_mask = src

    inputs: Dict[str, torch.Tensor] = assembler.batch_assemble(src_with_mask, tgt)
    labels = assembler.batch_assemble(src, tgt)
    inputs["lm_labels"] = make_lm_labels(tok, labels, inputs["input_ids"], shift=True)
    inputs["attention_mask"] = mix_attention_mask(
        inputs["attention_mask"], inputs["token_type_ids"]
    )
    return inputs


def load_run_file(path: str) -> Dict[str, np.ndarray]:
    output: Dict[str, List[Any]] = {"qid": [], "did": [], "label": []}
    with open(path, "r") as f:
        for line in f:
            splits = line.split()
            if len(splits) == 3:
                qid, did = splits[:2]
            elif len(splits) == 6:
                qid, did = splits[0], splits[2]
            else:
                assert False, f"Unrecognized format {line.rstrip()}"
            output["qid"].append(qid)
            output["did"].append(did)
            output["label"].append(0)
    array = {k: np.array(v) for k, v in output.items()}
    return array
