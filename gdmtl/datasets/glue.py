from typing import Any, Dict, Optional

import pandas as pd
import torch
from transformers import PreTrainedTokenizer

from .assembler import Assembler


class GlueDataset:
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        prefix_tokens: Optional[str],
        balance: bool = False,
        shuffle: bool = False,
    ):
        data = []
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data.append(line.rstrip("\n").split("\t"))
        df = pd.DataFrame(data[1:], columns=data[0])

        # QQP
        if "question1" in df.columns and "question2" in df.columns:
            df = df.rename(columns={"is_duplicate": "label"})
            df = df[["id", "question1", "question2", "label"]]

        # QNLI
        if "question" in df.columns and "sentence" in df.columns:
            if "label" in df.columns:
                df.loc[df["label"] == "entailment", "label"] = 1
                df.loc[df["label"] == "not_entailment", "label"] = 0
            df = df[["id", "question", "sentence", "label"]]

        if balance and "label" in df.columns:
            pos = df.loc[df["label"] == 1]
            neg = df.loc[df["label"] == 0]
            if len(pos) < len(neg):
                to_sample = len(neg) - len(pos)
                replace = to_sample > len(pos)
                pos.sample(n=to_sample, replace=replace, random_state=0)
                df = pd.concat([df, pos]).reset_index(drop=True)

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        df.loc[:, "label"] = df.loc[:, "label"].astype(int)
        self._df = df

        self._assembler = Assembler(
            tokenizer=tokenizer,
            max_length=max_length,
            prefix_token_ids=prefix_tokens,
            pad_to_max_length=False,
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if len(self._df.columns) == 4:
            id_, seq1, seq2, label = self._df.iloc[index]
        elif len(self._df.columns) == 3:
            id_, seq1, seq2 = self._df.iloc[index]

        outputs: Dict[str, Any] = self._assembler.batch_assemble([seq2], [seq1])
        outputs = {k: v[0] for k, v in outputs.items()}
        outputs.update(
            {
                "id": [str(id_)],
                # "qid1": [str(qid1)],
                # "qid2": [str(qid2)],
                "labels": torch.tensor([label]),
            }
        )

        assert all(
            v.dim() == 1 for v in outputs.values() if isinstance(v, torch.Tensor)
        )

        return outputs
