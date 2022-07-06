from argparse import ArgumentParser

from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.architectures.roberta.modeling_config import MONOLITHIC_HEAD_TYPES
from transformers_framework.models.base.mlm import BaseModelMLM
from transformers_framework.utilities.functional import index_multi_tensors


class BaseJointMLM(BaseModelMLM):

    def training_step(self, batch, *args):
        r"""
        Start by masking tokens some tokens.
        """
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        input_ids, labels = self.mlm(input_ids)

        # tokens_type_ids are automatically created by the model based on the config
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predictions, labels = index_multi_tensors(
            results.seq_class_logits.argmax(dim=-1), labels, positions=labels != IGNORE_IDX
        )

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('training/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('training/accuracy', self.train_mlm_acc(predictions, labels), on_epoch=True)

        return results.seq_class_loss

    def validation_step(self, batch, *args):
        r"""
        Start by masking tokens some tokens.
        """
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        input_ids, labels = self.mlm(input_ids)

        # tokens_type_ids are automatically created by the model based on the config
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predictions, labels = index_multi_tensors(
            results.seq_class_logits.argmax(dim=-1), labels, positions=labels != IGNORE_IDX
        )

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('validation/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('validation/accuracy', self.valid_mlm_acc(predictions, labels), on_epoch=True)

    def test_step(self, batch, *args):
        r"""
        Start by masking tokens some tokens.
        """
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        input_ids, labels = self.mlm(input_ids)

        # tokens_type_ids are automatically created by the model based on the config
        results = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        predictions, labels = index_multi_tensors(
            results.seq_class_logits.argmax(dim=-1), labels, positions=labels != IGNORE_IDX
        )

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('test/loss', results.seq_class_loss, on_epoch=True, prog_bar=True)
        self.log('test/accuracy', self.valid_mlm_acc(predictions, labels), on_epoch=True)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        super(BaseJointMLM, BaseJointMLM).add_model_specific_args(parser)
        parser.set_defaults(max_sequence_length=64)
        parser.add_argument('--head_type', type=str, required=True, choices=MONOLITHIC_HEAD_TYPES)
