from argparse import ArgumentParser
from typing import Any, List

from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
)
from transformers_lightning.language_modeling.masked_language_modeling import IGNORE_IDX

from transformers_framework.architectures.roberta.modeling_config import MONOLITHIC_HEAD_TYPES
from transformers_framework.models.base.mlm import BaseModelMLM
from transformers_framework.utilities import index_multi_tensors


class BaseJointMLMAndAS2(BaseModelMLM):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.train_acc = Accuracy()

        self.valid_acc = Accuracy()
        self.valid_map = RetrievalMAP()
        self.valid_mrr = RetrievalMRR()
        self.valid_p1 = RetrievalPrecision(k=1)
        self.valid_hr5 = RetrievalHitRate(k=5)
        self.valid_ndgc = RetrievalNormalizedDCG()

        self.valid_acc = Accuracy()
        self.valid_map = RetrievalMAP()
        self.valid_mrr = RetrievalMRR()
        self.valid_p1 = RetrievalPrecision(k=1)
        self.valid_hr5 = RetrievalHitRate(k=5)
        self.valid_ndgc = RetrievalNormalizedDCG()

    def training_step(self, batch, *args):
        r"""
        Start by masking tokens some tokens.
        """
        input_ids, attention_mask, labels, valid = (
            batch["input_ids"], batch["attention_mask"], batch["labels"], batch["valid"]
        )
        input_ids, mlm_labels = self.mlm(input_ids)
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=mlm_labels, class_labels=labels)

        # MLM part
        mlm_predictions = results.masked_lm_logits.argmax(dim=-1)
        mlm_predictions, mlm_labels = index_multi_tensors(
            mlm_predictions, mlm_labels, positions=mlm_labels != IGNORE_IDX
        )

        loss = results.masked_lm_loss + results.seq_class_loss

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('training/loss', loss, on_epoch=True, prog_bar=True)

        # MLM part
        self.log('training/mlm_loss', results.masked_lm_loss, on_epoch=True, prog_bar=True)
        self.log('training/mlm_accuracy', self.train_mlm_acc(mlm_predictions, mlm_labels), on_epoch=True)

        # Class part
        class_preds = results.seq_class_logits.argmax(dim=-1)
        class_preds, labels = index_multi_tensors(class_preds, labels, positions=valid)

        self.log('training/classification_loss', results.seq_class_loss, on_epoch=True)
        self.log('training/classification_accuracy', self.train_acc(class_preds, labels), on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        r"""
        Start by masking tokens some tokens.
        """
        input_ids, attention_mask, labels, keys, valid = (
            batch["input_ids"], batch["attention_mask"], batch['labels'], batch['keys'], batch['valid']
        )
        input_ids, mlm_labels = self.mlm(input_ids)
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=mlm_labels, class_labels=labels)

        # MLM part
        mlm_predictions = results.masked_lm_logits.argmax(dim=-1)
        mlm_predictions, mlm_labels = index_multi_tensors(
            mlm_predictions, mlm_labels, positions=mlm_labels != IGNORE_IDX
        )

        loss = results.masked_lm_loss + results.seq_class_loss

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('validation/loss', loss, on_epoch=True, prog_bar=True)

        # MLM part
        self.log('validation/mlm_loss', results.masked_lm_loss, on_epoch=True, prog_bar=True)
        self.log('validation/mlm_accuracy', self.train_mlm_acc(mlm_predictions, mlm_labels), on_epoch=True)

        # Class part
        keys = keys.unsqueeze(-1).expand_as(labels)
        logits, labels, keys = index_multi_tensors(results.seq_class_logits, labels, keys, positions=valid)

        preds = logits.argmax(dim=-1)
        scores = logits.softmax(dim=-1)[:, -1]

        self.valid_map.update(preds=scores, target=labels, indexes=keys)
        self.valid_mrr.update(preds=scores, target=labels, indexes=keys)
        self.valid_p1.update(preds=scores, target=labels, indexes=keys)
        self.valid_hr5.update(preds=scores, target=labels, indexes=keys)
        self.valid_ndgc.update(preds=scores, target=labels, indexes=keys)

        self.log('validation/classification_loss', results.seq_class_loss, on_epoch=True)
        self.log('validation/classification_accuracy', self.valid_acc(preds, labels), on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        r""" Just log metrics. """
        self.log('validation/map', self.valid_map, on_epoch=True)
        self.log('validation/mrr', self.valid_mrr, on_epoch=True)
        self.log('validation/p1', self.valid_p1, on_epoch=True)
        self.log('validation/hr5', self.valid_hr5, on_epoch=True)
        self.log('validation/ndcg', self.valid_ndgc, on_epoch=True)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        super(BaseJointMLMAndAS2, BaseJointMLMAndAS2).add_model_specific_args(parser)
        parser.set_defaults(max_sequence_length=64)
        parser.add_argument('--head_type', type=str, required=True, choices=MONOLITHIC_HEAD_TYPES)
