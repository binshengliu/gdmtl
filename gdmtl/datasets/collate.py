from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch


class PadCollate:
    def __init__(
        self,
        pad_token_id: int,
        pad_token_type_id: int,
        index_key: Optional[str] = "data_index",
    ):
        self._pad_token_id = pad_token_id
        self._pad_token_type_id = pad_token_type_id
        self._index_key = index_key

    def get_pad_id(self, key: str) -> int:
        if key.endswith("input_ids"):
            return self._pad_token_id
        elif key.endswith("attention_mask"):
            return 0
        elif key.endswith("token_type_ids"):
            return self._pad_token_type_id
        elif key.endswith("lm_labels"):
            return -100
        else:
            assert False, f"Unknown key {key}"

    def __call__(
        self, batch: Sequence[Mapping[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        from torch.utils.data.dataloader import default_collate  # type: ignore
        from irtools.pad import pad_to

        keys = {k for k, v in batch[0].items() if isinstance(v, torch.Tensor)}
        unalign = {k for k in keys if len(set(x[k].size() for x in batch)) > 1}
        sizes = {k: np.max([x[k].size() for x in batch], axis=0) for k in unalign}
        data = [
            {
                k: pad_to(v, sizes[k], self.get_pad_id(k)) if k in unalign else v
                for k, v in x.items()
            }
            for x in batch
        ]
        collated: Dict[str, torch.Tensor] = default_collate(data)
        if self._index_key:
            assert self._index_key not in collated, f"Key {self._index_key} exists"
            collated[self._index_key] = torch.arange(len(batch))
        return collated
