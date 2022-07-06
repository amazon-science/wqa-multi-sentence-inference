from argparse import ArgumentParser, Namespace
from typing import Callable

from pytorch_lightning import _logger as logger
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_warn
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers_lightning.adapters.super_adapter import SuperAdapter
from transformers_lightning.datamodules.adapter_datamodule import AdaptersDataModule
from transformers_lightning.datasets.iterable_dataset import TransformersIterableDataset
from transformers_lightning.datasets.map_dataset import TransformersMapDataset
from transformers_lightning.utils import collate_single_fn
from transformers_lightning.utils.inspectors import get_classes_from_module

from transformers_framework import adapters
from transformers_framework.adapters.transformer_adapter import TransformersAdapter
from transformers_framework.samplers import DistributedKeysSampler, KeysSampler
from transformers_framework.utilities.datamodules import STAGES_TO_NAMES


ADAPTER_CLASSES = get_classes_from_module(adapters, parent=TransformersAdapter)


class TransformersDataModule(AdaptersDataModule):
    r"""
    MultiFileDataModule implements some simple methods to check whether training, val or testing is required.
    This shoudl not directly instantiated.
    """

    def __init__(
        self,
        hyperparameters: Namespace,
        trainer: Trainer,
        collate_fn: Callable = collate_single_fn,
        tokenizer: PreTrainedTokenizer = None,
    ):
        super().__init__(hyperparameters, trainer, collate_fn=collate_fn)
        self.tokenizer = tokenizer
        self.adapter_class = ADAPTER_CLASSES[hyperparameters.adapter]

        self.train_adapter = self.get_adapter(TrainerFn.FITTING)
        self.valid_adapter = self.get_adapter(TrainerFn.VALIDATING)
        self.test_adapter = self.get_adapter(TrainerFn.TESTING)
        self.predict_adapter = self.get_adapter(TrainerFn.PREDICTING)

        assert not (hyperparameters.iterable is True and hyperparameters.keep_same_keys_close is True), (
            "cannot use `keep_same_keys_close` with `iterable` and vice-versa."
        )
        assert hyperparameters.keep_same_keys_close is False or hyperparameters.replace_sampler_ddp is False, (
            "when using `keep_same_keys_close` you must set `replace_sampler_ddp=False`"
        )

    # Optional, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: str = None):
        r""" Load datasets only if respective file is defined. """

        if stage is None:
            return

        if stage == TrainerFn.FITTING.value or stage == TrainerFn.VALIDATING.value:
            if self.do_train():
                self.train_dataset = self.load_dataset(TrainerFn.FITTING)
            if self.do_validation():
                self.valid_dataset = self.load_dataset(TrainerFn.VALIDATING)

        elif stage == TrainerFn.TESTING.value:
            if self.do_test():
                self.test_dataset = [self.load_dataset(TrainerFn.TESTING)]

        elif stage == TrainerFn.PREDICTING.value:
            if self.do_predict():
                self.predict_dataset = self.load_dataset(TrainerFn.PREDICTING)

    def get_adapter(self, stage: TrainerFn) -> SuperAdapter:
        r""" Return the adapter to use. """
        return self.adapter_class(
            self.hyperparameters, self.tokenizer, STAGES_TO_NAMES[stage], seed=self.trainer.current_epoch,
        )

    def load_dataset(self, stage: TrainerFn = None):
        r""" Load a dataset given the stage name. """
        logger.info(f"Loading {stage.value} dataset...")
        adapter = getattr(self, f"{STAGES_TO_NAMES[stage]}_adapter")
        dataset_class = TransformersIterableDataset if self.hyperparameters.iterable else TransformersMapDataset

        # map dataset must be told not to load everything in memory
        kwargs = {}
        if not self.hyperparameters.iterable:
            kwargs = dict(keep_in_memory=self.hyperparameters.force_load_dataset_in_memory)

        dataset = dataset_class(self.hyperparameters, adapter, self.trainer, **kwargs)
        rank_zero_info(
            f"{stage.value.capitalize()} dataset has length "
            f"{len(dataset) if not self.hyperparameters.iterable else 'inf'}"
        )
        return dataset

    def do_train(self):
        return self.train_adapter.is_active()

    def do_validation(self):
        return self.valid_adapter.is_active()

    def do_test(self):
        return self.test_adapter.is_active()

    def do_predict(self):
        return self.predict_adapter.is_active()

    def train_dataloader(self):
        r""" Return the training dataloader.
        If user requested keep_same_keys_close, we will provide a custom sampler to the dataloader. """
        if not self.do_train():
            return None

        if self.hyperparameters.keep_same_keys_close is True:
            # keep keys together only in training
            rank_zero_warn("Using custom keys sampler")
            sampler_cls = (
                DistributedKeysSampler if self.trainer.accelerator_connector.is_distributed else KeysSampler
            )
            sampler = sampler_cls(
                self.train_dataset,
                keep_same_keys_close=self.hyperparameters.keep_same_keys_close,
                shuffle=True,
            )
            return self.default_dataloader(self.train_dataset, self.hyperparameters.batch_size, sampler=sampler)

        if (
            self.hyperparameters.reload_dataloaders_every_n_epochs > 0
            and self.trainer.current_epoch > 0
            and self.trainer.current_epoch % self.hyperparameters.reload_dataloaders_every_n_epochs == 0
        ):
            rank_zero_warn("Reloading train dataset every epoch.")
            self.train_adapter = self.get_adapter(TrainerFn.FITTING)
            self.train_dataset = self.load_dataset(TrainerFn.FITTING)

        return self.default_dataloader(
            self.train_dataset,
            self.hyperparameters.batch_size,
            shuffle=not self.hyperparameters.iterable,
            prefetch_factor=self.hyperparameters.prefetch_factor,
        )

    @classmethod
    def add_datamodule_specific_args(cls, parser: ArgumentParser):
        super(TransformersDataModule, TransformersDataModule).add_datamodule_specific_args(parser)
        parser.add_argument('--adapter', type=str, required=True, choices=ADAPTER_CLASSES.keys())
        parser.add_argument(
            '--prefetch_factor', default=2, type=int, required=False, help='Number of examples to prepare in advance.'
        )
        parser.add_argument(
            '--keep_same_keys_close',
            action="store_true",
            help="Keep entries with same first together when shuffling, valid only for training.",
        )
        parser.add_argument(
            '--force_load_dataset_in_memory',
            action="store_true",
            help=(
                "Load whole dataset in memory even backed by pyarrow."
                " This may be usefull with transformations that change number of examples."
            )
        )
        tmp_hyperparameters, _ = parser.parse_known_args()
        adapter_class = ADAPTER_CLASSES[tmp_hyperparameters.adapter]
        for stage_name in STAGES_TO_NAMES.values():
            adapter_class.add_adapter_instance_specific_args(parser, stage_name=stage_name)
        adapter_class.add_adapter_specific_args(parser)

        
