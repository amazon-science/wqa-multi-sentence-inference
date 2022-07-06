from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class MaskedLMOutput(BaseModelOutput):
    r"""
    Base class for masked language modeling outputs.

    Args:
        masked_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
            `optional`, returned when :obj:`masked_lm_labels` is provided): Masked language modeling (MLM) loss.
        masked_lm_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    masked_lm_loss: Optional[torch.FloatTensor] = None
    masked_lm_logits: torch.FloatTensor = None


@dataclass
class SequenceClassificationOutput(BaseModelOutput):
    r"""
    Base class for sequence classification outputs.

    Args:
        seq_class_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`,
        `optional`, returned when :obj:`seq_class_labels` is provided):
            Sequence classification loss.
        seq_class_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Sequence classification logits (scores for each input example taken before SoftMax).
    """

    seq_class_loss: Optional[torch.FloatTensor] = None
    seq_class_logits: Optional[torch.FloatTensor] = None


@dataclass
class MaskedLMAndSequenceClassificationOutput(
    MaskedLMOutput, SequenceClassificationOutput
):
    r""" Masked language modeling + sequence classification. """
