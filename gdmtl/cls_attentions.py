from functools import reduce
from typing import List, Sequence, Tuple

import torch
from transformers import PreTrainedTokenizer


class ClsAttention:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        ignore_special_ids: bool = True,
        keep_pdf: bool = True,
    ):
        self._tokenizer = tokenizer
        self._ignore_special_ids = ignore_special_ids
        self._keep_pdf = keep_pdf
        indexes = list(self._tokenizer.get_vocab().values())
        word_begin = torch.tensor(
            [
                False if k.startswith("##") else True
                for k in self._tokenizer.get_vocab().keys()
            ]
        )
        self._word_begin = torch.zeros(len(word_begin), dtype=torch.bool)
        self._word_begin[indexes] = word_begin

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.Tensor, attentions: Sequence[torch.Tensor]
    ) -> Tuple[List[List[str]], List[List[float]]]:
        attentions = [x.mean(dim=1) for x in attentions]
        bsz, seq_len = input_ids.size()
        assert attentions[0].shape == (bsz, seq_len, seq_len)

        # [CLS] weights
        full = reduce(torch.matmul, attentions[::-1])[:, 0, :].cpu()

        input_ids = input_ids.cpu()
        batch_whole_words = []
        batch_attn_by_full = []
        for ids, attn in zip(input_ids, full):
            mask = ids != self._tokenizer.pad_token_id
            ids = ids[mask]
            attn = attn[mask]
            begin_idx = self._word_begin[ids].nonzero().view(-1)  # type: ignore
            begin_idx = torch.cat((begin_idx, torch.tensor([len(ids)])))
            sections = begin_idx[1:] - begin_idx[:-1]
            attn_by_full = torch.split(attn, sections.tolist())  # type: ignore
            attn_by_full = [x.sum().item() for x in attn_by_full]

            whole_words = torch.split(ids, sections.tolist())  # type: ignore
            whole_words = [self._tokenizer.decode(x) for x in whole_words]

            assert len(attn_by_full) == len(whole_words)
            batch_whole_words.append(whole_words)
            batch_attn_by_full.append(attn_by_full)
        return batch_whole_words, batch_attn_by_full
