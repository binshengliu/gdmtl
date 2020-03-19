import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_albert import AlbertMLMHead
from transformers.modeling_bert import (
    BertConfig,
    BertModel,
    BertOnlyMLMHead,
    BertPreTrainedModel,
)

log = logging.getLogger(__name__)


class MtlEncoderRanker(BertPreTrainedModel):  # type: ignore
    def __init__(self, config: BertConfig, **kwargs: Any):
        """The classification init is a super set of LM init"""
        super().__init__(config, **kwargs)
        self.config = config
        self.bert = BertModel(config=self.config)

        self.lm_head = BertOnlyMLMHead(self.config)
        self.lm_head.apply(self._init_weights)

        self.qa_head = BertOnlyMLMHead(self.config)
        self.qa_head.apply(self._init_weights)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.classifier.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = "summarizer",
        input_weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        """Versatile forward interface. By default it should behaves as an LM head so it's
           compatible with the `generate()` interface.

        labels: Labels for ranking.

        """
        model_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        if mode == "summarizer":
            lm_logits = self.lm_head(model_outputs[0])
            if labels is None:
                labels = kwargs.get("lm_labels", None)
            if labels is not None:
                if input_weights is None:
                    lm_loss = F.cross_entropy(
                        lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1)
                    )
                else:
                    lm_loss = F.cross_entropy(
                        lm_logits.view(-1, self.config.vocab_size),
                        labels.reshape(-1),
                        reduction="none",
                    )
                    # Weigh different examples
                    lm_loss = lm_loss.reshape(input_ids.size(0), -1)
                    lm_loss = lm_loss * input_weights.reshape(input_ids.size(0), 1)
                    lm_loss = lm_loss[labels != -100].mean()
                outputs = (lm_loss, lm_logits) + model_outputs[2:]
            else:
                outputs = (lm_logits,) + model_outputs[2:]
            return outputs
        elif mode == "qa":
            qa_logits = self.qa_head(model_outputs[0])
            if labels is not None:
                qa_loss = F.cross_entropy(
                    qa_logits.view(-1, self.config.vocab_size), labels.view(-1)
                )
                outputs = (qa_loss, qa_logits) + model_outputs[2:]
            else:
                outputs = (qa_logits,) + model_outputs[2:]
            return outputs
        elif mode == "ranker":
            rank_logits = self.classifier(self.dropout(model_outputs[1]))
            if labels is not None:
                loss = F.cross_entropy(
                    rank_logits.view(-1, self.config.num_labels), labels.view(-1)
                )
                outputs = (loss, rank_logits) + model_outputs[2:]
            else:
                outputs = (rank_logits,) + model_outputs[2:]
            return outputs
        else:
            assert False, f"Unknown mode {mode}"

    def get_output_embeddings(self) -> nn.Module:  # type: ignore
        return self.qa_head.predictions.decoder  # type: ignore

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Union[bool, torch.Tensor, None]]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": kwargs.get("token_type_ids"),
        }

    def shared_grads(self) -> Optional[torch.Tensor]:
        grads_list = []
        for name, params in self.bert.named_parameters():
            if name.startswith("pooler."):
                continue
            if params.requires_grad:
                if params.grad is not None:
                    grads_list.append(params.grad.flatten().cpu())
        if not grads_list:
            return None
        grads = torch.cat(grads_list)
        return grads

    def _init_weights(self, module: nn.Module) -> None:  # type: ignore
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_lm_head_cls(arch: str) -> nn.Module:  # type: ignore
        if arch.startswith("albert"):
            return AlbertMLMHead  # type: ignore
        else:
            return BertOnlyMLMHead  # type: ignore
