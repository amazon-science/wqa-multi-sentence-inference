import torch
from torchmetrics.classification import Accuracy, F1Score

from transformers_framework.models.base.base import BaseModel
from transformers_framework.utilities import shrink_batch


class BaseModelClassification(BaseModel):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.train_acc = Accuracy(num_classes=self.hyperparameters.num_labels)
        self.train_f1 = F1Score(num_classes=self.hyperparameters.num_labels, average=None)

        self.valid_acc = Accuracy(num_classes=self.hyperparameters.num_labels)
        self.valid_f1 = F1Score(num_classes=self.hyperparameters.num_labels, average=None)

        self.test_acc = Accuracy(num_classes=self.hyperparameters.num_labels)
        self.test_f1 = F1Score(num_classes=self.hyperparameters.num_labels, average=None)

    def training_step(self, batch, *args):
        r"""
        Just compute the loss and log it.
        """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch['labels']
        token_type_ids = batch.get('token_type_ids', None)

        input_ids, attention_mask, token_type_ids = shrink_batch(
            input_ids, attention_mask, token_type_ids, pad_token_id=self.tokenizer.pad_token_id
        )

        results = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        preds = results.logits.argmax(dim=-1)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('training/loss', results.loss, on_epoch=True, prog_bar=True)
        self.log('training/accuracy', self.train_acc(preds, labels), on_epoch=True, prog_bar=True)
        for class_id, value in enumerate(self.train_f1(preds, labels)):
            self.log(f'training/f1_class_{class_id}', value, on_epoch=True, prog_bar=False)

        return results.loss

    def validation_step(self, batch, *args):
        r"""
        Compute predictions and log retrieval results.
        """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch['labels']
        token_type_ids = batch.get('token_type_ids', None)

        input_ids, attention_mask, token_type_ids = shrink_batch(
            input_ids, attention_mask, token_type_ids, pad_token_id=self.tokenizer.pad_token_id
        )

        results = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        preds = results.logits.argmax(dim=-1)

        self.log('validation/accuracy', self.valid_acc(preds, labels), on_epoch=True, prog_bar=True)
        for class_id, value in enumerate(self.valid_f1(preds, labels)):
            self.log(f'validation/f1_class_{class_id}', value, on_epoch=True, prog_bar=False)

    def test_step(self, batch, *args):
        r"""
        Compute predictions and log retrieval results.
        """
        input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch['labels']
        token_type_ids = batch.get('token_type_ids', None)

        input_ids, attention_mask, token_type_ids = shrink_batch(
            input_ids, attention_mask, token_type_ids, pad_token_id=self.tokenizer.pad_token_id
        )

        results = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        preds = results.logits.argmax(dim=-1)

        self.log('test/accuracy', self.test_acc(preds, labels), on_epoch=True, prog_bar=True)
        for class_id, value in enumerate(self.test_f1(preds, labels)):
            self.log(f'test/f1_class_{class_id}', value, on_epoch=True, prog_bar=False)

    def predict_step(self, batch, *args):
        r"""
        Compute predictions.
        """
        input_ids, attention_mask, keys = batch["input_ids"], batch["attention_mask"], batch['keys']
        token_type_ids = batch.get('token_type_ids', None)

        input_ids, attention_mask, token_type_ids = shrink_batch(
            input_ids, attention_mask, token_type_ids, pad_token_id=self.tokenizer.pad_token_id
        )

        results = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        preds = results.logits.argmax(dim=-1)
        return {'preds': preds, 'keys': keys}

    def predict_epoch_end(self, predictions):
        r""" Receive a list of predictions and return a dict to write to files. """
        preds = torch.cat([o['preds'] for o in predictions], dim=0)
        keys = torch.cat([o['keys'] for o in predictions], dim=0)

        assert preds.shape == keys.shape
        return {'preds': preds.flatten(), 'keys': keys.flatten()}
