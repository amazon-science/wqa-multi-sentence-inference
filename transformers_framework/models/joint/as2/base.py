from argparse import ArgumentParser

import torch
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.roberta.modeling_config import MONOLITHIC_HEAD_TYPES
from transformers_framework.models.base.as2 import BaseModelAS2
from transformers_framework.utilities.functional import index_multi_tensors


class BaseJointAS2(BaseModelAS2):

    def training_step(self, batch, *args):
        r""" Just compute the loss and log it. """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch['labels']
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        preds = results.seq_class_logits.argmax(dim=-1)
        preds, labels = index_multi_tensors(preds, labels, positions=labels != IGNORE_IDX)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('training/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('training/accuracy', self.train_acc(preds, labels), on_epoch=True, prog_bar=True)

        return results.seq_class_loss

    def validation_step(self, batch, *args):
        r"""
        Compute predictions and log retrieval results.
        """
        input_ids, attention_mask, labels, keys, valid = (
            batch["input_ids"], batch["attention_mask"], batch['labels'], batch['keys'], batch['valid']
        )
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        keys = keys.unsqueeze(-1).expand_as(labels)
        logits, labels, keys = index_multi_tensors(results.seq_class_logits, labels, keys, positions=valid)

        preds = logits.argmax(dim=-1)
        scores = torch.softmax(logits, dim=-1)[:, -1]

        self.valid_map.update(preds=scores, target=labels, indexes=keys)
        self.valid_mrr.update(preds=scores, target=labels, indexes=keys)
        self.valid_p1.update(preds=scores, target=labels, indexes=keys)
        self.valid_hr5.update(preds=scores, target=labels, indexes=keys)
        self.valid_ndgc.update(preds=scores, target=labels, indexes=keys)

        self.log('validation/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('validation/accuracy', self.valid_acc(preds, labels), on_epoch=True, prog_bar=True)

    def test_step(self, batch, *args):
        r"""
        Compute predictions and log retrieval results.
        """
        input_ids, attention_mask, labels, keys, valid = (
            batch["input_ids"], batch["attention_mask"], batch['labels'], batch['keys'], batch['valid']
        )
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        keys = keys.unsqueeze(-1).expand_as(labels)
        logits, labels, keys = index_multi_tensors(results.seq_class_logits, labels, keys, positions=valid)

        preds = logits.argmax(dim=-1)
        scores = torch.softmax(logits, dim=-1)[:, -1]

        self.test_map.update(preds=scores, target=labels, indexes=keys)
        self.test_mrr.update(preds=scores, target=labels, indexes=keys)
        self.test_p1.update(preds=scores, target=labels, indexes=keys)
        self.test_hr5.update(preds=scores, target=labels, indexes=keys)
        self.test_ndgc.update(preds=scores, target=labels, indexes=keys)

        self.log('test/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('test/accuracy', self.test_acc(preds, labels), on_epoch=True, prog_bar=True)

    def predict_step(self, batch, *args):
        r""" Like test step but without metrics. """
        input_ids, attention_mask, keys, valid = (
            batch["input_ids"], batch["attention_mask"], batch['keys'], batch['valid']
        )
        results = self(input_ids=input_ids, attention_mask=attention_mask)

        keys = keys.unsqueeze(-1).expand_as(valid)
        logits, keys = index_multi_tensors(results.seq_class_logits, keys, positions=valid)
        scores = torch.softmax(logits, dim=-1)[..., -1]

        return {'scores': scores, 'keys': keys}

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        super(BaseJointAS2, BaseJointAS2).add_model_specific_args(parser)
        parser.set_defaults(max_sequence_length=64)
        parser.add_argument('--head_type', type=str, required=True, choices=MONOLITHIC_HEAD_TYPES)
