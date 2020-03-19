import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
)
from transformers.modeling_bart import BartClassificationHead, PretrainedBartModel

log = logging.getLogger(__name__)


class BartSumRank(
    BartForConditionalGeneration, BartForSequenceClassification  # type: ignore
):
    def __init__(self, config: BartConfig, **kwargs: Any):
        """The classification init is a super set of LM init"""
        PretrainedBartModel.__init__(self, config, **kwargs)
        self.model = BartModel(config)

        self.classification_head = BartClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)
        self.model._init_weights(self.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_cached_states: Optional[torch.Tensor] = None,
        lm_labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        rank_labels: Optional[torch.Tensor] = None,
        mode: str = "summarizer",
        **kwargs: Any,
    ) -> Any:
        """Versatile forward interface. By default it should behaves as an LM head so it's
           compatible with the `generate()` interface.

        lm_batch_mask: Used when the input_ids contain negative documents which are not
                       used for LM.

        rank_labels: Labels for ranking.

        """

        model_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if mode == "summarizer":
            lm_hidden = model_outputs[0]

            # LM head
            lm_logits = self.lm_head(lm_hidden)
            if lm_labels is not None:
                lm_loss = F.cross_entropy(
                    lm_logits.view(-1, self.config.vocab_size), lm_labels.reshape(-1)
                )
                outputs = (lm_loss, lm_logits) + model_outputs[1:]
            else:
                outputs = (lm_logits,) + model_outputs[1:]
            return outputs
        elif mode == "ranker":
            # Rank head
            rank_hidden = model_outputs[0]  # last hidden state
            bsz_idx = list(range(rank_hidden.size(0)))
            if decoder_attention_mask is not None:
                next_token_idx = decoder_attention_mask.sum(dim=1) - 1
            else:
                assert attention_mask is not None
                next_token_idx = attention_mask.sum(dim=1) - 1

            # Use next word prediction as sentence representation
            sentence_representation = rank_hidden[bsz_idx, next_token_idx]
            rank_logits = self.classification_head(sentence_representation)
            if rank_labels is not None:
                loss = F.cross_entropy(
                    rank_logits.view(-1, self.config.num_labels), rank_labels.view(-1)
                )
                outputs = (loss, rank_logits) + model_outputs[1:]
            else:
                outputs = (rank_logits,) + model_outputs[1:]
            return outputs
        else:
            assert False, f"Unknown mode {mode}"

    def shared_grads(self) -> Optional[torch.Tensor]:
        grads_list = []
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                if params.grad is not None:
                    grads_list.append(params.grad.flatten().cpu())
        if not grads_list:
            return None
        grads = torch.cat(grads_list)
        return grads
