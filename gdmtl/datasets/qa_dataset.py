import json
import logging
from typing import Any, Dict, Optional, Union

import ftfy
import numpy as np
from irtools.tqdmf import tqdmf
from more_itertools import first_true
from transformers import PreTrainedTokenizer
from unidecode import unidecode

from gdmtl.utils import local_rank

from .assembler import Assembler
from .data_typing import DatasetT
from .utils import make_targets_ntp_inputs

log = logging.getLogger(__name__)


class QADataset(DatasetT):
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizer,
        prefix: str,
        max_length: Optional[int],
        inference: bool = False,
    ):
        self._tokenizer = tokenizer
        self._prefix = prefix
        self._max_length = max_length
        self._inference = inference
        self._data = {}
        filtered = 0
        for line in tqdmf(
            path, desc=f"{local_rank()}: {path.split('/')[-1]}", position=local_rank()
        ):
            obj = json.loads(line)
            passage = first_true(obj["passages"], pred=lambda x: x["is_selected"] == 1)
            good_answer = obj["wellFormedAnswers"]
            if isinstance(good_answer, str):
                good_answer = good_answer.replace("[]", "")
            else:
                good_answer = min(
                    [x.replace("[]", "") for x in good_answer],
                    key=lambda x: len(x.split()),
                )

            if passage is None or obj["answers"] == ["No Answer Present."]:
                filtered += 1
                continue

            self._data[str(obj["query_id"])] = [
                obj["query"].strip(),
                passage["passage_text"].strip(),
                obj["answers"][0].strip(),
                good_answer.strip(),
            ]
        self._index = list(self._data.keys())
        log.info(f"QA dataset {path}: {len(self._data)} examples; {filtered} invalid.")
        log.info("Missing qids will be replaced with random ones.")
        self._assembler = Assembler(
            tokenizer=tokenizer,
            max_length=max_length,
            prefix_token_ids=prefix,
            pad_to_max_length=False,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        qid = self._index[index]
        return self.by_qid(qid)

    def by_qid(self, qid: Union[int, str]) -> Dict[str, Any]:
        def clean(x: str) -> str:
            return unidecode(ftfy.fix_text(x, fix_entities=False))  # type: ignore

        qid = str(qid)

        if qid not in self._data:
            qid = np.random.choice(list(self._data.keys()), 1)[0]

        assert isinstance(qid, str)
        query, passage, answer, goodanswer = [clean(x) for x in self._data[qid]]
        if goodanswer:
            answer = goodanswer

        src_seq = [passage + " [SEP] " + query]
        if self._inference:
            inputs: Dict[str, Any] = self._assembler.batch_assemble(src_seq)

            inputs["lm_labels"] = self._tokenizer.encode(
                answer, add_special_tokens=False, return_tensors="pt"
            )
            inputs = {k: v[0] for k, v in inputs.items()}
            assert all(v.dim() == 1 for k, v in inputs.items())
        else:
            inputs = make_targets_ntp_inputs(
                self._assembler, self._tokenizer, src_seq, [answer]
            )
            inputs = {k: v[0] for k, v in inputs.items()}
            assert all(v.dim() == 1 for k, v in inputs.items() if k != "attention_mask")
            assert inputs["attention_mask"].dim() == 2

        inputs["qids"] = [qid]

        return inputs
