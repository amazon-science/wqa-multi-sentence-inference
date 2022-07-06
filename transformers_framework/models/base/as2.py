from typing import Any, List

import torch
from torchmetrics.classification import Accuracy
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
)
from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.models.base.base import BaseModel
from transformers_framework.utilities import shrink_batch
from transformers_framework.utilities.functional import index_multi_tensors


class BaseModelAS2(BaseModel):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.train_acc = Accuracy()

        self.valid_acc = Accuracy()
        self.valid_map = RetrievalMAP()
        self.valid_mrr = RetrievalMRR()
        self.valid_p1 = RetrievalPrecision(k=1)
        self.valid_hr5 = RetrievalHitRate(k=5)
        self.valid_ndgc = RetrievalNormalizedDCG()

        self.test_acc = Accuracy()
        self.test_map = RetrievalMAP()
        self.test_mrr = RetrievalMRR()
        self.test_p1 = RetrievalPrecision(k=1)
        self.test_hr5 = RetrievalHitRate(k=5)
        self.test_ndgc = RetrievalNormalizedDCG()

    def training_step(self, batch, *args):
        r""" Just compute the loss and log it. """
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
        preds, labels = index_multi_tensors(preds, labels, positions=labels != IGNORE_IDX)

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('training/loss', results.loss, on_epoch=True, prog_bar=True)
        self.log('training/accuracy', self.train_acc(preds, labels), on_epoch=True, prog_bar=True)
        return results.loss

    def validation_step(self, batch, *args):
        r""" Compute predictions and log retrieval results. """
        input_ids, attention_mask, labels, keys = (
            batch["input_ids"], batch["attention_mask"], batch['labels'], batch['keys']
        )
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
        scores = torch.softmax(results.logits, dim=-1)[..., -1]
        preds, scores, labels = index_multi_tensors(preds, scores, labels, positions=labels != IGNORE_IDX)

        self.valid_map.update(preds=scores, target=labels, indexes=keys)
        self.valid_mrr.update(preds=scores, target=labels, indexes=keys)
        self.valid_p1.update(preds=scores, target=labels, indexes=keys)
        self.valid_hr5.update(preds=scores, target=labels, indexes=keys)
        self.valid_ndgc.update(preds=scores, target=labels, indexes=keys)

        self.log('validation/accuracy', self.valid_acc(preds, labels), on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        r""" Just log metrics. """
        self.log('validation/map', self.valid_map, on_epoch=True)
        self.log('validation/mrr', self.valid_mrr, on_epoch=True)
        self.log('validation/p1', self.valid_p1, on_epoch=True)
        self.log('validation/hr5', self.valid_hr5, on_epoch=True)
        self.log('validation/ndcg', self.valid_ndgc, on_epoch=True)

    def test_step(self, batch, *args):
        r""" Compute predictions and log retrieval results. """
        input_ids, attention_mask, labels, keys = (
            batch["input_ids"], batch["attention_mask"], batch['labels'], batch['keys']
        )
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
        scores = torch.softmax(results.logits, dim=-1)[..., -1]
        preds, scores, labels = index_multi_tensors(preds, scores, labels, positions=labels != IGNORE_IDX)

        self.test_map.update(preds=scores, target=labels, indexes=keys)
        self.test_mrr.update(preds=scores, target=labels, indexes=keys)
        self.test_p1.update(preds=scores, target=labels, indexes=keys)
        self.test_hr5.update(preds=scores, target=labels, indexes=keys)
        self.test_ndgc.update(preds=scores, target=labels, indexes=keys)

        self.log('test/accuracy', self.test_acc(preds, labels), on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        r""" Just log metrics. """
        self.log('test/map', self.test_map, on_step=False, on_epoch=True)
        self.log('test/mrr', self.test_mrr, on_step=False, on_epoch=True)
        self.log('test/p1', self.test_p1, on_step=False, on_epoch=True)
        self.log('test/hr5', self.test_hr5, on_step=False, on_epoch=True)
        self.log('test/ndcg', self.test_ndgc, on_step=False, on_epoch=True)

    def predict_step(self, batch, *args):
        r""" Like test step but without metrics and labels. """
        input_ids, attention_mask, keys = (
            batch["input_ids"], batch["attention_mask"], batch['keys']
        )
        token_type_ids = batch.get('token_type_ids', None)

        input_ids, attention_mask, token_type_ids = shrink_batch(
            input_ids, attention_mask, token_type_ids, pad_token_id=self.tokenizer.pad_token_id
        )

        results = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        scores = torch.softmax(results.logits, dim=-1)[:, 1]    # take predictions on pos class
        return {'scores': scores, 'keys': keys}

    def predict_epoch_end(self, predictions):
        r""" Receive a list of predictions and return a List of scores to write to a file. """
        scores = torch.cat([o['scores'] for o in predictions], dim=0)
        keys = torch.cat([o['keys'] for o in predictions], dim=0)

        assert scores.shape == keys.shape
        return {'scores': scores.flatten(), 'keys': keys.flatten()}
