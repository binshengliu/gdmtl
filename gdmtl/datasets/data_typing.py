from typing import TYPE_CHECKING, Dict

import torch
from torch.utils.data import DataLoader, Dataset

# https://github.com/python/mypy/issues/5264
if TYPE_CHECKING:
    DataLoaderT = DataLoader[Dict[str, torch.Tensor]]
    DatasetT = Dataset[Dict[str, torch.Tensor]]
else:
    DataLoaderT = DataLoader
    DatasetT = Dataset
