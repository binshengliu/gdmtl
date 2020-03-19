#!/usr/bin/env python
# coding: utf-8
import inspect
import io
import logging
import re
from collections import abc
from contextlib import redirect_stdout
from math import ceil
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pytorch_lightning as pl
import torch
from irtools.merge_dict import merge
from more_itertools import flatten
from omegaconf import DictConfig
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from torch.optim import Optimizer  # type: ignore
from torch.utils.data import DataLoader
from transformers import AdamW, AutoConfig, AutoTokenizer

from gdmtl.datasets import (
    DataLoaderT,
    DatasetT,
    PadCollate,
    TsvCollection,
    get_dataset_cls,
)
from gdmtl.logger import ModelLogger
from gdmtl.losses import RankHingeLoss, UncertaintyLoss, UnlikelihoodLoss
from gdmtl.metrics import TrecEval
from gdmtl.models import BartSumRank, MtlEncoderRanker
from gdmtl.utils import get_adamw_params, get_linear_schedule_with_warmup_and_minimum

# Configured by pytorch-lightning
logger = logging.getLogger(__name__)


class TrainMixin(pl.LightningModule):  # type: ignore
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._cfg = cfg
        config = AutoConfig.from_pretrained(
            cfg.arch, revision=cfg.revision, **cfg.model_config
        )
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.arch, revision=cfg.revision)
        if "bart" in cfg.arch:
            self._model = BartSumRank.from_pretrained(
                cfg.arch, config=config, revision=cfg.revision
            )
        elif "bert" in cfg.arch:
            self._model = MtlEncoderRanker.from_pretrained(
                cfg.arch, config=config, revision=cfg.revision
            )
        else:
            assert f"Unsupported arch: {cfg.arch}"

        if hasattr(self._cfg, "balance") and self._cfg.balance == "uncertainty":
            for k in ["summarizer", "orig", "var", "disc", "rl", "qa"]:
                key = f"weight_{k}"
                if isinstance(self._cfg[key], float) and self._cfg[key] > 0:
                    logger.info(f"Change {key}: {self._cfg[key]} -> uncertainty")
                    self._cfg[key] = "uncertainty"

        self._uncertainty = UncertaintyLoss(min=cfg.min_uncertainty)

        self._train_loaders: List[DataLoaderT] = []
        self._epoch_idx = 0
        self._running_loss: Dict[str, TensorRunningAccum] = {}

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> Any:
        spec = inspect.getfullargspec(self._model.forward)
        if spec.varargs is None and spec.varkw is None:
            inputs = {k: v for k, v in inputs.items() if k in set(spec.args)}
        outputs = self._model(**inputs)

        return outputs

    def on_train_start(self) -> None:
        self.log_params_histogram()

    def training_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        results = self._training_step(batch, batch_idx)

        bsz = batch["sum_input_ids"].size(0)
        results.update(self._seq_and_tokens(batch))
        results.update(
            {
                "qids": batch["qids"],
                "dnos": batch["dnos"],
                "bsz": batch["sum_input_ids"].new_tensor([bsz]),
            }
        )

        if (
            self._cfg.log_embeddings > 0
            and self.trainer.proc_rank == 0
            and batch_idx % self._cfg.log_embeddings == 0
        ):
            self.log_embeddings()

        return results

    def training_step_end(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        log_interaction = (
            self._cfg.log_grad_interaction > 0
            and self.trainer.proc_rank == 0
            and self.trainer.global_step % self._cfg.log_grad_interaction == 0
        )

        if log_interaction:
            named_losses = {
                k.split("_weighted_loss")[0]: v.mean()
                for k, v in outputs.items()
                if "weighted_loss" in k
            }
            self.log_grad_interaction(named_losses)

        self._update_running_loss(outputs)
        loss = torch.stack(
            [v.mean() for k, v in outputs.items() if "weighted_loss" in k]
        ).sum()

        log = {f"train/{k}": v.mean() for k, v in outputs.items() if "loss" in k}
        log.update(
            {f"train/{k}": v.mean() for k, v in outputs.items() if "uncertainty" in k}
        )
        log.update(
            {f"train/{k}": v.mean() for k, v in outputs.items() if "seq_len" in k}
        )
        log.update({f"train/{k}": v.sum() for k, v in outputs.items() if "tokens" in k})
        log.update({"train/loss": loss, "train/bsz": outputs["bsz"].sum()})
        for i, lr in enumerate(self._scheduler.get_last_lr()):
            lr = loss.new_tensor(lr)
            log[f"train/lr_{i}"] = lr

        scalars: Dict[str, Any] = {
            "qids": list(flatten(outputs["qids"])),
            "dids": list(flatten(outputs["dnos"])),
            **log,
        }
        self.log_step(scalars)
        return {"loss": loss, "log": log}

    def on_after_backward(self) -> None:
        if self._cfg.log_grad_histogram > 0 and (
            self.trainer.global_step % self._cfg.log_grad_histogram == 0
        ):
            self.log_grad_histgram()

        if self._cfg.log_grad_norm > 0 and (
            self.trainer.global_step % self._cfg.log_grad_norm == 0
        ):
            self.log_grad_norm()

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self._uncertainty.on_before_zero_grad(optimizer)

    @torch.no_grad()
    def training_epoch_end(self, outputs: Any) -> Dict[str, Dict[str, torch.Tensor]]:
        self.log_euclidean()
        self.log_params_histogram()

        return {}

    def prepare_data(self) -> None:
        if self._cfg.test_only:
            return
        logger.info("Prepare data. This may take a while.")
        self._prepare_train_dataloader()

    def train_dataloader(self) -> DataLoaderT:
        logger.debug(f"Train data loader for epoch {self._epoch_idx}")
        return self._train_loaders[self._epoch_idx]

    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        params = get_adamw_params(self._model)
        self._optimizer = AdamW(params, lr=self._cfg.lr, eps=1e-8)

        if self._cfg.test_only:
            return ([self._optimizer], [])

        self._optimizer.add_param_group(
            {"params": self._uncertainty.parameters(), "lr": self._cfg.uncertainty_lr}
        )

        epoch_steps = [
            ceil(len(self._train_loaders[i])) for i in range(self._cfg.max_epochs)
        ]
        warmup_steps = sum(epoch_steps[: self._cfg.warmup_epochs])
        work_steps = sum(
            epoch_steps[self._cfg.warmup_epochs :][: self._cfg.work_epochs]
        )
        total_steps = sum(epoch_steps)
        logger.info("Epoch steps: {}".format(" ".join(map(str, epoch_steps))))
        logger.info(f"Warmup steps: {warmup_steps}")
        logger.info(f"Work steps: {work_steps}")
        logger.info(f"Cooldown steps: {total_steps - work_steps - warmup_steps}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Lr: {self._cfg.lr}")
        logger.info(f"Min lr: {self._cfg.min_lr}")
        min_ratio = self._cfg.min_lr / self._cfg.lr
        self._scheduler = get_linear_schedule_with_warmup_and_minimum(
            self._optimizer,
            warmup_steps,
            total_steps,
            min_ratio=min_ratio,
            num_work_steps=work_steps,
        )

        return (
            [self._optimizer],
            [{"scheduler": self._scheduler, "interval": "step", "frequency": 1}],
        )

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        fout = io.StringIO()
        with redirect_stdout(fout):
            super().optimizer_step(*args, **kwargs)
        for line in fout.getvalue().splitlines():
            match = re.search(r"reducing loss scale to (.*)", line)
            if match:
                self._loss_scale = float(match[1])
            logger.debug(line)

    def on_epoch_end(self) -> None:
        logger.debug(f"Epoch {self._epoch_idx} ends")
        self._epoch_idx += 1

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        bar_dict: Dict[str, Union[int, str]] = super().get_progress_bar_dict()
        bar_dict.pop("v_num", None)
        if "ranker_loss_orig" in self._running_loss:
            ranker_running_loss = self._running_loss["ranker_loss_orig"].mean()
            if ranker_running_loss is not None:
                bar_dict["ranker"] = f"{ranker_running_loss:.3f}"

        if hasattr(self, "_loss_scale"):
            bar_dict["scale"] = f"{self._loss_scale:.0f}"

        return bar_dict

    def _training_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        results = {}

        if (
            self._cfg.weight_summarizer == "uncertainty"
            or self._cfg.weight_summarizer > 0
        ):
            gen_results = self._training_step_gen(batch, batch_idx)
            results.update(gen_results)

        if self._cfg.weight_qa == "uncertainty" or self._cfg.weight_qa > 0:
            qa_results = self._training_step_qa(batch, batch_idx)
            results.update(qa_results)

        if self._cfg.weight_orig == "uncertainty" or self._cfg.weight_orig > 0:
            rank_results = self._training_step_rank(batch, batch_idx)
            results.update(rank_results)

        return results

    def _training_step_rank(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        bsz, gsz, _ = batch["rank_input_ids"].size()
        results = {}

        batch = {
            k: v.view(bsz * gsz, -1) if k.startswith("rank") else v
            for k, v in batch.items()
        }
        inputs = {
            "input_ids": batch["rank_input_ids"],
            "attention_mask": batch["rank_attention_mask"],
            "token_type_ids": batch.get("rank_token_type_ids"),
            "decoder_input_ids": batch.get("rank_decoder_input_ids"),
            "decoder_attention_mask": batch.get("rank_decoder_attention_mask"),
            "mode": "ranker",
            "output_attentions": False,
        }
        if batch_idx == 0:
            self._log_one_example(inputs)

        should_log_attention = (
            self._cfg.log_attentions > 0
            and self.trainer.proc_rank == 0
            and batch_idx % self._cfg.log_attentions == 0
        )
        if should_log_attention:
            inputs["output_attentions"] = True
            outputs = self(inputs)
            if len(outputs) == 2:
                ranker_logits, attentions = outputs
            elif len(outputs) == 4:
                ranker_logits, decoder_attentions, _, attentions = outputs
            else:
                assert False, f"Unkonwn output format of length {len(outputs)}."
        else:
            ranker_logits = self(inputs)[0]

        ranker_loss = RankHingeLoss(num_neg=gsz - 1)(ranker_logits.view(-1))

        if self._cfg.weight_orig == "uncertainty":
            ranker_weighted_loss = self._uncertainty(ranker_loss, "ranker")
            results["ranker_uncertainty"] = self._uncertainty["ranker"]
        else:
            ranker_weighted_loss = ranker_loss * self._cfg.weight_orig

        results["ranker_loss_orig"] = ranker_loss
        results["ranker_weighted_loss_orig"] = ranker_weighted_loss

        if should_log_attention:
            logger.debug("Log attentions")
            self.log_attentions(batch, attentions, ranker_logits)
        return results

    def _training_step_gen(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        bsz, gsz, _ = batch["sum_input_ids"].size()
        results = {}
        inputs: Mapping[str, Any] = batch
        if self._cfg.unlikelihood > 0.0:
            inputs = {
                k: v.view(-1, *(v.size()[2:]))
                for k, v in inputs.items()
                if k == "lm_labels" or k.startswith("sum")
            }
        else:
            inputs = {
                k: v[:, 0, ...]
                for k, v in inputs.items()
                if k == "lm_labels" or k.startswith("sum")
            }

        inputs = {
            "input_ids": inputs["sum_input_ids"],
            "attention_mask": inputs["sum_attention_mask"],
            "token_type_ids": inputs.get("sum_token_type_ids"),
            "decoder_input_ids": inputs.get("sum_decoder_input_ids"),
            "decoder_attention_mask": inputs.get("sum_decoder_attention_mask"),
            "lm_labels": (None if self._cfg.unlikelihood > 0 else inputs["lm_labels"]),
            "input_weights": inputs.get("sum_input_weights"),
            "mode": "summarizer",
        }
        if not self._cfg.weigh_qgen:
            inputs.pop("input_weights", None)

        if batch_idx == 0:
            self._log_one_example(inputs)

        outputs = self(inputs)
        if self._cfg.unlikelihood == 0.0:
            summarizer_loss, summarizer_logits = outputs[:2]
        else:
            loss_func = UnlikelihoodLoss(gsz - 1, self._cfg.unlikelihood)
            summarizer_loss, likely_loss, unlikely_loss = loss_func(
                outputs[0], batch["lm_labels"].view(bsz * gsz, -1)
            )
            results["summarizer_likely_loss"] = likely_loss
            results["summarizer_unlikely_loss"] = unlikely_loss

        if self._cfg.weight_summarizer == "uncertainty":
            summarizer_weighted_loss = self._uncertainty(summarizer_loss, "qgen")
            results["summarizer_uncertainty"] = self._uncertainty["qgen"]
        else:
            summarizer_weighted_loss = summarizer_loss * self._cfg.weight_summarizer

        results["summarizer_loss"] = summarizer_loss
        results["summarizer_weighted_loss"] = summarizer_weighted_loss
        return results

    def _training_step_qa(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        results = {}
        inputs = {
            "input_ids": batch["qa_input_ids"],
            "attention_mask": batch["qa_attention_mask"],
            "token_type_ids": batch.get("qa_token_type_ids"),
            "decoder_input_ids": batch.get("qa_decoder_input_ids"),
            "decoder_attention_mask": batch.get("qa_decoder_attention_mask"),
            "labels": batch["qa_lm_labels"],
            "mode": "qa",
        }

        if batch_idx == 0:
            self._log_one_example(inputs)

        qa_loss = self(inputs)[0]

        if self._cfg.weight_qa == "uncertainty":
            qa_weighted_loss = self._uncertainty(qa_loss, "qa")
            results["qa_uncertainty"] = self._uncertainty["qa"]
        else:
            qa_weighted_loss = qa_loss * self._cfg.weight_qa

        results["qa_loss"] = qa_loss
        results["qa_weighted_loss"] = qa_weighted_loss
        return results

    def _generate_from_sum_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        disable_special_ids: bool = True,
        sample: bool = False,
        disable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # bsz, seq_len, vocab_size
        assert logits.dim() == 3
        # bart-large and bart-large-cnn have different vocab_size but there is some
        # inconsistency in the implementation.
        if disable_special_ids:
            specials_ids = self._tokenizer.all_special_ids
            specials_ids = [
                x for x in specials_ids if x < self._model.config.vocab_size
            ]
            logits[..., specials_ids] = float("-inf")

        if disable is not None:
            disable[disable == -100] = self._tokenizer.pad_token_id
            logits.scatter_(
                dim=-1, index=disable[:, :, None], src=logits.new_tensor(float("-inf"))
            )

        if sample:
            bsz, seq_len, vocab_size = logits.size()
            query_input_ids = torch.multinomial(
                logits.view(-1, vocab_size).softmax(dim=-1), 1
            )
            query_input_ids = query_input_ids.reshape(bsz, seq_len)
        else:
            query_input_ids = logits.argmax(-1)
        query_input_ids[~mask.bool()] = self._tokenizer.pad_token_id
        return query_input_ids

        # The last token is eos. Omit it.
        # eos_indexes = mask.sum(dim=1, keepdims=True) - 1  # type:ignore
        # query_input_ids.scatter_(1, eos_indexes, self._tokenizer.pad_token_id)
        # attention_mask = mask.scatter(1, eos_indexes, 0)
        # return query_input_ids, attention_mask

    @property
    def collection(self) -> TsvCollection:
        if not hasattr(self, "_collection"):
            self._collection = TsvCollection(self._cfg.collection)
        return self._collection

    def _prepare_train_dataloader(self) -> None:
        """Re-sample training negative documents in every epoch."""
        train_queries = TsvCollection(self._cfg.train_query)

        dataset_cls = get_dataset_cls(self._cfg.train_data_cls)
        train_sets = dataset_cls.create(
            data=self._cfg.train_data,
            tokenizer=self._tokenizer,
            query_col=train_queries,
            doc_col=self.collection,
            num_dup=self._cfg.num_dup,
            num_neg=self._cfg.num_neg,
            max_length=self._cfg.max_len,
            decoder_start_token_id=self._model.config.decoder_start_token_id,
            src_max_length=self._cfg.src_max_len,
            tgt_max_length=self._cfg.tgt_max_len,
            epochs=self._cfg.max_epochs,
            summarizer_prefix_token_ids=self._cfg.summarize_prefix,
            rank_prefix_token_ids=self._cfg.rank_prefix,
            pad_to_max_length=self._cfg.train_pad_to_max_length,
            qa_data=self._cfg.qa_train_path,
            qa_prefix=self._cfg.qa_prefix,
            mask_whole_word_prob=self._cfg.mask_whole_word_prob,
            mask_qgen_query=self._cfg.mask_qgen_query,
            mask_query_from_passage=self._cfg.mask_query_from_passage,
            min_rel_for_qgen=self._cfg.train_qgen_min_rel,
        )
        logger.info(f"Train number of examples: {len(train_sets[0])}")

        train_loaders = []
        for epoch_idx, train_set in enumerate(train_sets):
            dataloader = DataLoader(
                train_set,
                num_workers=self._cfg.train_data_workers,
                collate_fn=PadCollate(
                    self._tokenizer.pad_token_id, self._tokenizer.pad_token_type_id
                ),
                batch_size=self._cfg.train_bsz,
                shuffle=self._cfg.train_shuffle,
            )
            train_loaders.append(dataloader)
        self._train_loaders = train_loaders

    def _seq_and_tokens(
        self, batch: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        for k, v in batch.items():
            if not k.endswith("_input_ids"):
                continue
            key = "seq_len_" + k.replace("_input_ids", "")
            value = torch.tensor(v.size(-1), dtype=torch.float, device=self._device)
            outputs[key] = value

            key = "tokens_" + k.replace("_input_ids", "")
            value = torch.tensor(v.numel(), dtype=torch.float, device=self._device)
            outputs[key] = value
        return outputs

    def _update_running_loss(self, outputs: Mapping[str, torch.Tensor]) -> None:
        for k, v in outputs.items():
            if "weighted_loss" in k:
                continue
            if "loss" not in k:
                continue
            self._running_loss.setdefault(k, TensorRunningAccum(window_length=20))
            self._running_loss[k].append(v.mean())

    def _log_one_example(self, batch: Mapping[str, torch.Tensor]) -> None:
        buffers = ["Example:"]
        for key, value in batch.items():
            if value is None:
                continue
            if isinstance(value, (bool, float, int, str, list)):
                buffers.append(f"{key}: {value}")
            elif isinstance(value, torch.Tensor):
                if value.dim() <= 1:
                    buffers.append(f"{key}: {value}")
                elif value.dim() == 2:
                    buffers.append(f"{key}: {value[0]}")
                else:
                    buffers.append(f"{key}: {value[0, 0]}")

                if "input_ids" in key:
                    sent = self._tokenizer.decode(value[0])
                    buffers.append(f"{key}: {sent}")
        logger.debug("\n".join(buffers))


class ValMixin(pl.LightningModule):  # type: ignore
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if not hasattr(self, "_treceval"):
            self._treceval: Dict[str, List[TrecEval]] = {}
        self._treceval["valid"] = []
        for i in range(len(cfg.valid_qrels)):
            trec_eval = TrecEval(
                cfg.valid_metrics[i],
                self._cfg.valid_qrels[i],
                options=cfg.valid_options[i],
            )
            self._treceval["valid"].append(trec_eval)

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        rank_inputs = {
            "input_ids": batch["rank_input_ids"],
            "attention_mask": batch["rank_attention_mask"],
            "token_type_ids": batch.get("rank_token_type_ids"),
            "decoder_input_ids": batch.get("rank_decoder_input_ids"),
            "decoder_attention_mask": batch.get("rank_decoder_attention_mask"),
            "mode": "ranker",
        }
        y_pred = self(rank_inputs)[0]
        return {
            "qids": batch["qids"],
            "dnos": batch["dnos"],
            "logits": y_pred,
            "labels": batch["labels"],
        }

    def validation_step_end(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # return tensor_to_numpy(outputs)  # type: ignore
        return outputs  # type: ignore

    def validation_epoch_end(
        self,
        outputs: Union[
            Sequence[Sequence[Mapping[str, torch.Tensor]]],
            Sequence[Mapping[str, torch.Tensor]],
        ],
    ) -> Dict[str, Dict[str, float]]:
        return self._eval_epoch_end("valid", outputs)

    def _eval_epoch_end(
        self,
        stage: str,
        outputs_seq: Union[
            Sequence[Sequence[Mapping[str, torch.Tensor]]],
            Sequence[Mapping[str, torch.Tensor]],
        ],
    ) -> Dict[str, Dict[str, float]]:
        if not isinstance(outputs_seq[0], abc.Sequence):
            outputs_seq = [outputs_seq]  # type: ignore

        outputs: Dict[str, Dict[str, float]] = {}
        for idx, one in enumerate(outputs_seq):
            assert isinstance(one, abc.Sequence)
            merge(outputs, self._evaluate_rank(self._epoch_idx, one, stage, idx))

        return outputs

    def val_dataloader(self) -> Union[List[DataLoaderT], DataLoaderT]:
        return self._eval_dataloader("valid")

    def _eval_dataloader(self, stage: str) -> Union[List[DataLoaderT], DataLoaderT]:
        if not hasattr(self, f"_{stage}_sets"):

            def one_set(eval_query: str, eval_data: str) -> DatasetT:
                query = TsvCollection(eval_query)

                dataset_cls = get_dataset_cls(self._cfg.eval_data_cls)
                eval_set = dataset_cls(
                    data=eval_data,
                    tokenizer=self._tokenizer,
                    query_col=query,
                    doc_col=self.collection,
                    max_length=self._cfg.max_len,
                    src_max_length=self._cfg.src_max_len,
                    tgt_max_length=self._cfg.tgt_max_len,
                    decoder_start_token_id=self._model.config.decoder_start_token_id,
                    pad_to_max_length=self._cfg[f"{stage}_pad_to_max_length"],
                    sort=self._cfg[f"{stage}_sort"],
                    rank_prefix_token_ids=self._cfg.rank_prefix,
                    first_seq="passage",
                )
                return eval_set

            if isinstance(self._cfg[f"{stage}_query"], str):
                query_list = [self._cfg[f"{stage}_query"]]
                data_list = [self._cfg[f"{stage}_data"]]
            else:
                query_list = self._cfg[f"{stage}_query"]
                data_list = self._cfg[f"{stage}_data"]
            eval_sets = []
            for eval_query, data in zip(query_list, data_list):
                eval_sets.append(one_set(eval_query, data))
            setattr(self, f"_{stage}_sets", eval_sets)

        dataloaders = [
            DataLoaderT(
                x,
                num_workers=self._cfg[f"{stage}_data_workers"],
                collate_fn=PadCollate(
                    self._tokenizer.pad_token_id, self._tokenizer.pad_token_type_id
                ),
                batch_size=self._cfg[f"{stage}_bsz"],
            )
            for x in getattr(self, f"_{stage}_sets")
        ]
        if len(dataloaders) == 1:
            return dataloaders[0]
        else:
            return dataloaders

    def _evaluate_rank(
        self,
        epoch_idx: int,
        outputs: Sequence[Mapping[str, torch.Tensor]],
        mode: str,
        idx: int = 0,
    ) -> Dict[str, Dict[str, float]]:
        gather = {k: torch.cat([x[k] for x in outputs], dim=0) for k in outputs[0]}
        qids = gather["qids"].cpu().squeeze(1).tolist()
        dnos = gather["dnos"].cpu().squeeze(1).tolist()
        logits = gather["logits"].cpu().squeeze(1).tolist()
        labels = gather.get("labels")
        if labels is not None:
            labels = labels.cpu().squeeze(1).tolist()

        filename = self._write_run(
            f"{mode}{idx}", epoch_idx, qids, dnos, logits, labels
        )

        # No eval labels
        if idx >= len(self._treceval[mode]):
            return {}

        metric = self._treceval[mode][idx](filename)
        metric = {canonize_name(k): v for k, v in metric.items()}

        log = {f"{mode}{idx}/{k}": v for k, v in metric.items()}

        # Add an alias without '/' for checkpoint to use as filename.
        main_metric = canonize_name(self._cfg[f"{mode}_metrics"][idx][0])
        log[f"{main_metric}-{idx}"] = metric[main_metric]
        progress_bar = {f"{main_metric}-{idx}": metric[main_metric]}
        return {"log": log, "progress_bar": progress_bar}

    def _write_run(
        self,
        stage: str,
        epoch: Optional[int],
        qids: Sequence[int],
        dnos: Sequence[int],
        logits: Sequence[float],
        labels: Optional[Sequence[int]] = None,
    ) -> str:
        if labels is None:
            labels = [None] * len(qids)

        by_qids: MutableMapping[str, List[Tuple[str, float, Any]]] = {}
        for qid, did, logit, label in zip(qids, dnos, logits, labels):
            by_qids.setdefault(qid, []).append((did, logit, label))

        for qid in by_qids:
            by_qids[qid] = sorted(by_qids[qid], key=lambda x: x[1], reverse=True)

        rank_log: Dict[str, int] = {}
        if epoch is not None:
            filename = f"{stage}-epoch={epoch}-run.run"
        else:
            filename = f"{stage}-run.run"
        dedup = set()
        with open(filename, "w") as f:
            for qid in by_qids:
                for dno, logit, label in by_qids[qid]:
                    if (qid, dno) in dedup:
                        continue
                    dedup.add((qid, dno))
                    rank_log.setdefault(qid, 1)
                    line = f"{qid} Q0 {dno} {rank_log[qid]} {logit} indri"
                    if label is not None:
                        line += f" # {label:1.0f}"
                    f.write(line + "\n")
                    rank_log[qid] += 1
        logger.debug(f"Run file saved in {filename}")
        return filename


def canonize_name(metric: str) -> str:
    metric = re.sub(r"recip_rank_cut[._]", "mrr", metric)
    metric = re.sub(r"ndcg_cut[._]", "ndcg", metric)
    metric = re.sub(r"_cut[._]", "", metric)
    return metric


class TestMixin(pl.LightningModule):  # type: ignore
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if not hasattr(self, "_treceval"):
            self._treceval: Dict[str, List[TrecEval]] = {}
        self._treceval["test"] = []
        for i in range(len(cfg.test_qrels)):
            trec_eval = TrecEval(
                cfg.test_metrics[i],
                self._cfg.test_qrels[i],
                options=cfg.test_options[i],
            )
            self._treceval["test"].append(trec_eval)

    def test_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx, *args, **kwargs)  # type: ignore

    def test_step_end(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> Dict[str, np.ndarray]:
        return self.validation_step_end(outputs)  # type: ignore

    def test_epoch_end(
        self,
        outputs_seq: Union[
            Sequence[Sequence[Mapping[str, np.ndarray]]],
            Sequence[Mapping[str, np.ndarray]],
        ],
    ) -> Dict[str, Dict[str, float]]:
        return self._eval_epoch_end("test", outputs_seq)  # type: ignore

    def test_dataloader(self) -> List[DataLoaderT]:
        return self._eval_dataloader("test")  # type: ignore


class Trainer(ModelLogger, TestMixin, ValMixin, TrainMixin):  # type: ignore
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
