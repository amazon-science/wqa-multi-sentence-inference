from argparse import ArgumentParser

import torch

from transformers_framework.architectures.roberta.modeling_config import MONOLITHIC_HEAD_TYPES
from transformers_framework.models.base.classification import BaseModelClassification


class BaseJointFactChecking(BaseModelClassification):

    def training_step(self, batch, *args):
        r""" Just compute the loss and log it. """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch['labels']
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = results.seq_class_logits.argmax(dim=-1)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('training/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('training/accuracy', self.train_acc(preds, labels), on_epoch=True, prog_bar=True)
        for class_id, value in enumerate(self.train_f1(preds, labels)):
            self.log(f'training/f1_class_{class_id}', value, on_epoch=True, prog_bar=False)
        return results.seq_class_loss

    def validation_step(self, batch, *args):
        r"""
        Compute predictions and log retrieval results.
        """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch['labels']
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = results.seq_class_logits.argmax(dim=-1)

        self.log('validation/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('validation/accuracy', self.valid_acc(preds, labels), on_epoch=True, prog_bar=True)
        for class_id, value in enumerate(self.valid_f1(preds, labels)):
            self.log(f'validation/f1_class_{class_id}', value, on_epoch=True, prog_bar=False)

    def test_step(self, batch, *args):
        r"""
        Compute predictions and log retrieval results.
        """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch['labels']
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = results.seq_class_logits.argmax(dim=-1)

        self.log('test/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('test/accuracy', self.test_acc(preds, labels), on_epoch=True, prog_bar=True)
        for class_id, value in enumerate(self.test_f1(preds, labels)):
            self.log(f'test/f1_class_{class_id}', value, on_epoch=True, prog_bar=False)

    def predict_step(self, batch, *args):
        r"""
        Compute predictions.
        """
        input_ids, attention_mask, keys = batch["input_ids"], batch["attention_mask"], batch['keys']
        results = self(input_ids=input_ids, attention_mask=attention_mask)
        preds = results.seq_class_logits.argmax(dim=-1)
        return {'preds': preds, 'keys': keys}

    def predict_epoch_end(self, predictions):
        r""" Receive a list of predictions and return a dict to write to files. """
        preds = torch.cat([o['preds'] for o in predictions], dim=0)
        keys = torch.cat([o['keys'] for o in predictions], dim=0)
        assert preds.shape == keys.shape
        return {'preds': preds.flatten(), 'keys': keys.flatten()}

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        super(BaseJointFactChecking, BaseJointFactChecking).add_model_specific_args(parser)
        parser.set_defaults(max_sequence_length=64)
        parser.add_argument('--head_type', type=str, required=True, choices=MONOLITHIC_HEAD_TYPES)
