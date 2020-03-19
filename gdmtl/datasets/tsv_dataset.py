from __future__ import annotations

from typing import Dict, KeysView, Union

import ftfy
from irtools.tqdmf import tqdmf
from unidecode import unidecode

from gdmtl.utils import local_rank


class TsvCollection:
    def __init__(self, path: str, clean_text: bool = True):
        assert path.endswith(".tsv"), f"Unkonwn formath {path}. Use .tsv."
        self._output: Dict[str, str] = {}
        for line in tqdmf(
            path, desc=f"{local_rank()}: {path.split('/')[-1]}", position=local_rank()
        ):
            splits = line.rstrip("\n").split("\t", maxsplit=1)
            self._output[splits[0]] = splits[1]
        self._clean_text = clean_text
        self._clean: Dict[str, str] = {}
        self._len_cache: Dict[str, int] = {}

    def __getitem__(self, key: Union[int, str]) -> str:
        key = str(key)
        if not self._clean_text:
            return self._output[key]

        if key not in self._clean:
            self._clean[key] = unidecode(
                ftfy.fix_text(self._output[key], fix_entities=False)
            )
        return self._clean[key]

    def __contains__(self, el: str) -> bool:
        return el in self._output

    def __len__(self) -> int:
        return len(self._output)

    def keys(self) -> KeysView[str]:
        return self._output.keys()

    def tokens(self, key: Union[int, str]) -> int:
        key = str(key)

        if key not in self._len_cache:
            self._len_cache[key] = len(self._output[key].split())
        return self._len_cache[key]
