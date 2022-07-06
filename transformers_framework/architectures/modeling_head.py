import math
from typing import Tuple

import torch
from torch import nn
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.modeling_output import SequenceClassificationOutput


class ClassificationHead(nn.Module):
    r""" Head for sentence-level classification tasks. """

    def __init__(self, config, hidden_size: int = None, num_labels: int = None):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        num_labels = num_labels if num_labels is not None else config.num_labels

        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        features = self.dropout(features)
        features = self.dense(features)
        features = torch.tanh(features)
        features = self.dropout(features)
        features = self.out_proj(features)
        return features


class JointClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size * 2 if config.head_type == "AE_k" else config.hidden_size

        self.config = config
        self.cls_positions = [self.config.sentence_msl * i for i in range(self.config.k + 1)]
        self.classifier = ClassificationHead(config, hidden_size=hidden_size)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)

    def forward(self, hidden_state: torch.Tensor, labels: torch.Tensor) -> SequenceClassificationOutput:
        r"""
        Args:
            hidden_state: the last hidden_state of some model with shape (batch_size, (k + 1) * seq_len, hidden_size)
            labels: the labels for every candidate with shape (batch_size) or (batch_size, k)

        Return:
            the loss and the logits of shape (batch_size, num_labels) or (batch_size, k, num_labels).
        """

        if self.config.head_type == "IE_1":
            hidden_state = hidden_state[:, self.cls_positions[0], :]
        elif self.config.head_type == "AE_1":
            hidden_state = hidden_state[:, self.cls_positions, :].sum(dim=1)
        elif self.config.head_type == "IE_k":
            hidden_state = hidden_state[:, self.cls_positions[1:], :]
        else: # "AE_k"
            question_hidden_states = hidden_state[:, [self.cls_positions[0]], :]
            candidates_hidden_states = hidden_state[:, self.cls_positions[1:], :]
            question_hidden_states = question_hidden_states.expand_as(candidates_hidden_states)
            hidden_state = torch.cat([question_hidden_states, candidates_hidden_states], dim=2)

        logits = self.classifier(hidden_state)

        loss = None
        if labels is not None:
            assert 1 <= labels.dim() <= 2, "labels must be of shape (batch_size) or (batch_size, k)"

            if self.config.head_type == "IE_1":
                assert labels.dim() == 1, "IE_1 classification head needs labels of shape (batch_size)"
            elif self.config.head_type == "AE_1":
                assert labels.dim() == 1, "AE_1 classification head needs labels of shape (batch_size)"
            elif self.config.head_type == "IE_k":
                assert labels.dim() == 2, "IE_k classification head needs labels of shape (batch_size, k)"
            else: # "AE_k"
                assert labels.dim() == 2, "AE_k classification head needs labels of shape (batch_size, k)"

            loss = self.loss_fct(logits.view(-1, self.config.num_labels), labels.flatten())

        return SequenceClassificationOutput(seq_class_loss=loss, seq_class_logits=logits)
