#!/usr/bin/env python
# coding: utf-8
import logging
import os

import hydra
import pytorch_lightning as pl
import transformers  # noqa: let transformers initialize its logger so hydra can configure it
from omegaconf import DictConfig, OmegaConf

from gdmtl.trainer import Trainer
from gdmtl.utils import (
    local_rank,
    log_versions,
    seed_all,
    suppress_trivial_warnings,
    to_trainer_args,
)

# Configured by pytorch-lightning
log = logging.getLogger(__name__)


@hydra.main(config_path="../conf/gdmtl")  # type:ignore
def main(cfg: DictConfig) -> None:
    suppress_trivial_warnings()
    log_versions()
    log.info(f"Local rank: {local_rank()}")
    seed_all(cfg.seed)

    assert cfg.unlikelihood == 0.0, "Unlikelihood is not properly supported yet"

    if cfg.load_checkpoint:
        model = Trainer.load_from_checkpoint(cfg.load_checkpoint, cfg=cfg, strict=False)
    else:
        model = Trainer(cfg)

    cb = pl.callbacks.ModelCheckpoint(
        filepath="checkpoints/{epoch}-{mrr10-0:.3f}-{ndcg10-1:.3f}",
        monitor="mrr10-0",
        save_top_k=-1,
        mode="max",
    )

    trainer = pl.Trainer(checkpoint_callback=cb, **to_trainer_args(cfg))
    model.num_gpus = trainer.num_gpus

    log.info(os.getcwd())
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.test_only:
        trainer.test(model)
    else:
        trainer.fit(model)

    log.info(os.getcwd())


if __name__ == "__main__":
    main()
