from __future__ import annotations

import inspect
import logging
import os
import re
from collections import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
from more_itertools import always_iterable
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer  # type:ignore
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup_and_minimum(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
    min_ratio: float = 0.0,
    num_work_steps: int = 0,
) -> _LRScheduler:
    """Create a schedule with a learning rate that decreases linearly
    after linearly increasing during a warmup period and stabilizing
    for a working period.

    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_work_steps:
            return 1.0
        cooldown = num_training_steps - num_warmup_steps - num_work_steps
        return max(
            min_ratio,
            float(num_training_steps - current_step) / float(max(1, cooldown)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)  # type: ignore


TPath = TypeVar("TPath", Sequence[str], str)


def get_path(data_dirs: Sequence[str], name: Optional[TPath]) -> Optional[TPath]:
    def resolve_path_for_one(one_name: str) -> str:
        for one in data_dirs:
            path = Path(one, one_name).expanduser()
            if path.is_file():
                return str(path)
        raise ValueError(f"Can't find {name} in {data_dirs}")

    if isinstance(name, str):
        return resolve_path_for_one(name)
    elif isinstance(name, abc.Sequence):
        return [resolve_path_for_one(x) for x in name]
    elif name is None:
        return None
    raise ValueError(f"Can't find {name} in {data_dirs}")


def fix_path(cfg: DictConfig) -> DictConfig:
    from hydra.utils import to_absolute_path as abs_path

    cfg.data_dirs = list(always_iterable(cfg.data_dirs))
    num_dirs = len(cfg.data_dirs)
    cfg.data_dirs = [
        abs_path(os.path.expanduser(cfg.data_dirs[i])) for i in range(num_dirs)
    ]
    path_patterns = [
        r"^(train|valid|test|cv)_(qrels|query|data)$",
        r"^collection$",
        r"qa_(train|valid|test)_path$",
        r"^(load_checkpoint|resume_from_checkpoint)$",
        r"^(train|valid|test)_(src|tgt)_path$",
    ]

    for key in cfg.keys():
        if any(re.search(pat, key) for pat in path_patterns):
            orig = cfg[key]
            cfg[key] = get_path(cfg.data_dirs, cfg[key])
            logger.debug(f"Resolve {key}: {orig} -> {cfg[key]}")

    return cfg


def get_adamw_params(model: torch.nn.Module[torch.Tensor]) -> List[Dict[str, Any]]:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 5e-5,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def seed_all(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def suppress_trivial_warnings() -> None:
    import os
    import warnings

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings(action="ignore", message="Seems like")
    warnings.filterwarnings(action="ignore", message="Did not find hyperparameters")
    warnings.filterwarnings(action="ignore", message="Detected call of")


def log_versions() -> None:
    import transformers
    import pytorch_lightning as pl
    import subprocess
    from hydra.utils import get_original_cwd as cwd

    try:
        cmd = f"git -C {cwd()} describe --dirty --always --long"
        myver = subprocess.check_output(cmd, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        logger.error("Please run from the root of the code repo.")
        exit(1)

    logger.info(f"HEAD: {myver}")
    logger.info(f"PyTorch: {torch.__version__}")  # type: ignore
    logger.info(f"CUDA: {torch.version.cuda}")  # type: ignore
    logger.info(f"Transformers: {transformers.__version__}")
    logger.info(f"PyTorch-Lightning: {pl.__version__}")


def local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0


def to_trainer_args(cfg: DictConfig) -> Dict[str, Any]:
    dict_cfg = OmegaConf.to_container(cfg)
    assert isinstance(dict_cfg, abc.Mapping)

    if hasattr(pl.Trainer.__init__, "__wrapped__"):
        args = set(inspect.getfullargspec(pl.Trainer.__init__.__wrapped__).args)
    else:
        args = set(inspect.getfullargspec(pl.Trainer.__init__).args)
    dict_args = {k: v for k, v in dict_cfg.items() if k in args}
    return dict_args
