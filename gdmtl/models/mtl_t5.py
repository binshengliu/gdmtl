import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import T5Config, T5ForConditionalGeneration, T5Model

log = logging.getLogger(__name__)


class T5SumRank(T5ForConditionalGeneration):  # type: ignore
    def __init__(self, config: T5Config, **kwargs: Any):
        """The classification init is a super set of LM init"""
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_past_key_value_states: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        lm_labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        mode: str = "summarizer",
    ) -> Any:
        """A wrapper of summarizer and ranker. It is exposed as a summarizer
        because we want to take advantage of generate() function.
        """

        assert mode in ["ranker", "summarizer"], f"Unknown mode {mode}"
        if mode == "summarizer":
            return self.summarize(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                decoder_past_key_value_states=decoder_past_key_value_states,
                use_cache=use_cache,
                lm_labels=lm_labels,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                head_mask=head_mask,
            )
        elif mode == "ranker":
            assert input_ids is not None
            return self.rank(
                input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels
            )
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def summarize(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_past_key_value_states: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        lm_labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        return T5ForConditionalGeneration.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_past_key_value_states=decoder_past_key_value_states,
            use_cache=use_cache,
            lm_labels=lm_labels,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            head_mask=head_mask,
        )

    def rank(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        t5_outputs = T5Model.forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        next_token_logits = t5_outputs[0][:, -1, :]
        logits = self.dropout(next_token_logits)
        logits = self.classifier(logits)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()  # type:ignore
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs  # type:ignore

        return outputs
