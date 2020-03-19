import fcntl
import logging
import re
from functools import reduce
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence

import jsonlines
import pytorch_lightning as pl
import torch
from irtools.pytorch_recipes import tensor_to_primitive
from more_itertools import flatten
from omegaconf import DictConfig

from gdmtl.cls_attentions import ClsAttention
from gdmtl.models import MtlEncoderRanker

log = logging.getLogger(__name__)


class ModelLogger(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._viz = ClsAttention(self._tokenizer)
        self._initial_params = self._get_model_params()

    def log_params_histogram(self) -> None:
        # Log params
        for name, params in self._get_model_params().items():
            self.logger.experiment.add_histogram(
                f"model_params/{name}", params, self.trainer.global_step
            )

    def log_grad_histgram(self) -> None:
        for name, grads in self._get_model_grads().items():
            self.logger.experiment.add_histogram(
                f"model_grads/{name}", grads, self.trainer.global_step
            )

    def log_attentions(
        self,
        batch: Mapping[str, torch.Tensor],
        attentions: Sequence[torch.Tensor],
        logits: torch.Tensor,
    ) -> None:
        log.debug("Log attentions")
        words, weights = self._viz(batch["rank_input_ids"], attentions)
        # figs, pdfs = self._viz.to_figure(words, weights)
        qids = list(flatten(batch["qids"]))
        qids = [qids[x] for x in batch["data_index"].tolist()]
        dnos = list(flatten(batch["dnos"]))
        dnos = [dnos[x] for x in batch["data_index"].tolist()]
        rel = batch["data_index"].new_zeros(batch["data_index"].size(0))
        rel[:: (self._cfg.num_neg + 1)] = 1

        step = self.trainer.global_step
        with open("attentions.tsv", "a") as ftsv:
            # This function may be executed in multithreading/multiprocessing context.
            fcntl.flock(ftsv, fcntl.LOCK_EX)
            for qid, dno, r, l, wd, wt in zip(
                qids, dnos, rel.tolist(), logits.view(-1).tolist(), words, weights
            ):
                sent = " ".join(flatten(zip(wd, map(str, wt))))
                ftsv.write(f"{step}\t{qid}\t{dno}\t{r}\t{l:.4f}\t{sent}\n")
            fcntl.flock(ftsv, fcntl.LOCK_UN)

    def log_grad_norm(self) -> None:
        for name, grads in self._get_model_grads().items():
            self.logger.experiment.add_scalar(
                f"grad_2_norm_total/{name}",
                grads.norm(2),  # type: ignore
                self.trainer.global_step,
            )

    def log_embeddings(self) -> None:
        device = next(self._model.parameters()).device
        words = "rank [CLS] [SEP]".split()
        ids2 = self._tokenizer.convert_tokens_to_ids(words)
        # ids = ids[ids != self._tokenizer.pad_token_id].unique()  # type: ignore
        tensor = torch.tensor(ids2, device=device, dtype=torch.long)
        metadata = [(x, self.trainer.global_step) for x in words]
        metadata_header = ["word", "step"]
        embeddings = self._model.model.embeddings.word_embeddings(tensor)
        self.logger.experiment.add_embedding(
            mat=embeddings.cpu(),
            metadata=metadata,
            tag="word_embeddings",
            global_step=self.trainer.global_step,
            metadata_header=metadata_header,
        )

    def log_euclidean(self) -> None:
        current_parameters = self._get_model_params()
        for k, v in current_parameters.items():
            dist = torch.dist(self._initial_params[k], v)
            self.logger.experiment.add_scalar(
                f"euclidean/{k}",
                dist,
                self.trainer.global_step,
            )

    def log_grad_interaction(self, losses: Mapping[str, torch.Tensor]) -> None:
        if len(losses) <= 1:
            log.debug("One loss found. Skip grad interaction analysis.")
            return

        grads = {}
        for key, loss in losses.items():
            loss.backward(retain_graph=True)  # type: ignore
            model_grad = self._model.shared_grads()
            if model_grad is None:
                self._optimizer.zero_grad()
                continue
            grads[key] = model_grad
            self._optimizer.zero_grad()

            norm = grads[key].norm()
            name = f"grad_interaction/norm-{key}"
            self.logger.experiment.add_scalar(name, norm, self.trainer.global_step)

        if len(grads) <= 1:
            log.debug("One task grads found. Skip grad interaction analysis.")
            return

        if len(set(x.numel() for x in grads.values())) > 1:
            log.debug(
                f"Grad num inconsistent {[x.numel() for x in grads.values()]}. "
                "Skip grad interaction analysis."
            )
            return

        log.debug(f"Log grad interactions {list(grads.keys())}")
        for num in range(2, len(grads) + 1):
            for keys in combinations(grads.keys(), num):
                norm = reduce(lambda x, y: x + y, (grads[k] for k in keys)).norm()
                name = "grad_interaction/norm/" + "-".join(keys)
                self.logger.experiment.add_scalar(name, norm, self.trainer.global_step)

                sign_tensor = torch.stack([grads[x] for x in keys], dim=1).sign()
                sign_tensor = sign_tensor == sign_tensor[:, 0:1]
                sign_tensor = sign_tensor.all(dim=1)
                ratio = sign_tensor.sum().float() / sign_tensor.numel()
                name = "grad_interaction/direction/" + "-".join(keys)
                self.logger.experiment.add_scalar(name, ratio, self.trainer.global_step)

        for key in grads.keys():
            others = list(set(grads.keys()) - {key})
            key_sign = grads[key].sign()
            # Change of direction from one other task
            for other in others:
                changed = (grads[key] + grads[other]).sign() != key_sign
                self.logger.experiment.add_scalar(
                    f"grad_interaction/impact/{other}_on_{key}",
                    changed.sum().float() / changed.numel(),
                    self.trainer.global_step,
                )

            # Change of direction from two other tasks
            combined = reduce(lambda x, y: x + y, grads.values())
            changed = key_sign != combined.sign()
            others_name = "_".join(others)
            self.logger.experiment.add_scalar(
                f"grad_interaction/impact/{others_name}_on_{key}",
                changed.sum().float() / changed.numel(),
                self.trainer.global_step,
            )

    def log_step(self, log: Dict[Any, Any]) -> None:
        raw_log = {"step": self.trainer.global_step}
        raw_log.update(tensor_to_primitive(log))
        with jsonlines.open("scalars.jsonl", "a") as f:
            f.write(raw_log)

    def _get_model_params(self) -> Dict[str, torch.Tensor]:
        parameters = {}
        for children, module in self._model.named_children():
            module_param = [
                p.cpu().reshape(-1) for p in module.parameters() if p.requires_grad
            ]
            if not module_param:
                continue
            parameters[children] = torch.cat(module_param, dim=0)
        return parameters

    def _get_model_grads(self) -> Dict[str, torch.Tensor]:
        grads = {}
        for children, module in self._model.named_children():
            module_grads = [
                p.grad.cpu().reshape(-1)
                for p in module.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if not module_grads:
                continue
            grads[children] = torch.cat(module_grads, dim=0)
        return grads

    def _replay_transition(
        self, model: MtlEncoderRanker, input_ids: torch.Tensor, hiddens: Any
    ) -> None:
        embs = model.model.embeddings.word_embeddings.weight
        assert embs.dim() == 2

        # Increase one dimension to broadcast on bsz dim
        embs = embs[None, ...]
        assert embs.dim() == 3

        found = torch.empty(0, dtype=torch.long, device=input_ids.device)
        for layer_out in hiddens:
            assert layer_out.dim() == 3
            assert layer_out.shape[:2] == input_ids.shape
            layer_out = layer_out.view(-1, 1, layer_out.size(-1))
            for one in torch.split(layer_out, 32):  # type: ignore
                estimation = torch.norm(embs - one, dim=-1).argmin(  # type: ignore
                    dim=-1
                )
                found = torch.cat((found, estimation))
        found = found.view((len(hiddens),) + input_ids.size())


def normalize_param_name(name: str) -> str:
    patterns = [r"(.*\.layers?\.\d+)", r"(.*_head)"]
    norm_name = ""
    for pat in patterns:
        match = re.match(pat, name)
        if match:
            norm_name = match[0]
            break
    if norm_name == "":
        norm_name = re.sub(r"(\.weight|\.bias)$", "", name)
    return norm_name
