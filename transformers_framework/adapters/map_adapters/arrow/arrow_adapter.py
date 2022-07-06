from argparse import ArgumentParser, Namespace
from json import JSONDecodeError
from typing import Dict

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from pytorch_lightning import _logger as logger
from pytorch_lightning.utilities import rank_zero_warn
from transformers import PreTrainedTokenizer

from transformers_framework.adapters.map_adapters.map_adapter import MapAdapter
from transformers_framework.utilities.processors import data_processor
from transformers_framework.utilities.structures import DataSample


def load_dataset_from_disk(path: str, keep_in_memory: bool = False, split: str = None) -> DatasetDict:
    r""" Load both Dataset's dumps and json folders transparently from disk. """
    try:
        res = load_from_disk(path, keep_in_memory=keep_in_memory)
        if split is not None and split != "-":
            res = res[split]
    except FileNotFoundError:
        try:
            res = load_dataset('json', data_dir=path, keep_in_memory=keep_in_memory)['train']
            if split is not None and split != "-":
                rank_zero_warn(
                    "Jsonl dataset does not require a split, just use `--splits -`."
                    " For this run I will set `--splits -` for you."
                )
        except JSONDecodeError:
            logger.error(
                f"Could not load dataset from {path}. "
                f"Make sure this path is a valid folder containing jsonl files or a dataset dump."
            )
            exit(1)
    return res


class ArrowAdapter(MapAdapter):
    r""" Superclass of Arrow File readers, which implements filtering on scores and limits. """

    def __init__(
        self,
        hyperparameters: Namespace,
        tokenizer: PreTrainedTokenizer,
        stage_name: str,
        seed: int = 0,
    ) -> None:
        super().__init__(hyperparameters, tokenizer, stage_name, seed=seed)

        if self.is_active():
            self.data = self.load_data()

    def load_data(self):
        r""" Load data from disk first parsing input parameters. This method should be protected by `is_active`. """
        filepaths = self.hyperparameters[f'{self.stage_name}_filepaths']
        splits = self.hyperparameters[f'{self.stage_name}_splits']

        if len(splits) == 1 and len(filepaths) > 1:
            splits = splits * len(filepaths)

        assert len(splits) == len(filepaths), (
            "You must provide a single split for every dataset or a split for every dataset"
        )

        rank_zero_warn(f"Loading datasets from disk{' and concatenating' if len(filepaths) > 1 else ''}...")
        data = concatenate_datasets([
            load_dataset_from_disk(filepath, keep_in_memory=self.hyperparameters.keep_in_memory, split=split)
            for split, filepath in zip(splits, filepaths)
        ])

        for field in self.hyperparameters.field_names:
            for f in field.split(":"):
                assert f in data.column_names, (
                    f"column {f} was not found among available dataset's columns {data.column_names}"
                )
        return data

    def is_active(self) -> bool:
        return self.hyperparameters[f'{self.stage_name}_filepaths'] is not None

    def check(self, sample: Dict, idx: int):
        if any(v is None for v in sample.values()):
            if not hasattr(self, "already_logged_warning"):
                self.already_logged_warning = True
                rank_zero_warn(
                    f"Sample {sample} with id {idx} seems incomplete. Will not log warning like this anymore."
                )

    def __getitem__(self, idx) -> Dict:
        r""" Get dict of data at a given position. """
        sample = self.data[idx]
        if self.hyperparameters.generate_key is True and ('key' not in sample or sample['key'] is None):
            sample['key'] = idx
        self.check(sample, idx)  # check sample is valid and raise warning the first time
        return self.transformations(sample)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for idx, sample in enumerate(self.data):
            self.check(sample, idx)  # check sample is valid and raise warning
            yield self.transformations(sample)

    def preprocess_line(self, sample: DataSample) -> Dict:
        r"""
        Process a line. The structure of each line is exactly
        the same returned by the __iter__ method. Here you should do data preparation
        for the actual model being trained. This is a good place to do tokenization,
        padding and so on.
        """
        return data_processor(
            sample,
            tokenizer=self.tokenizer,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            chain=self.hyperparameters.chain,
        )

    @staticmethod
    def add_adapter_specific_args(parser: ArgumentParser):
        super(ArrowAdapter, ArrowAdapter).add_adapter_specific_args(parser)
        parser.add_argument('--keep_in_memory', action="store_true", help="Read whole Dataset into memory.")
        parser.add_argument('--max_sequence_length', required=True, type=int, help="Model max sequence length")
        parser.add_argument(
            '--chain', action="store_true", help="Whether to chain sentences when encoded separately"
        )
        parser.add_argument(
            '--field_names',
            required=True,
            nargs='+',
            help="Names of the fields of the input data to use for training. Use : to concatenate field together."
        )
        parser.add_argument(
            '--key_name',
            required=False,
            default=None,
            help="Name of the key field"
        )
        parser.add_argument(
            '--label_name',
            required=False,
            default=None,
            help="Name of the label field"
        )
        parser.add_argument('--generate_key', action="store_true", help="Use dataset indexes as keys.")

    @staticmethod
    def add_adapter_instance_specific_args(parser: ArgumentParser, stage_name: str):
        super(ArrowAdapter, ArrowAdapter).add_adapter_instance_specific_args(parser, stage_name=stage_name)
        parser.add_argument(
            f'--{stage_name}_filepaths',
            type=str,
            required=False,
            default=None,
            nargs='+',
            help=f"Path to {stage_name} dataset dump",
        )
        parser.add_argument(
            f'--{stage_name}_splits',
            type=str,
            required=False,
            default=[stage_name],
            nargs='+',
            help="The dataset split to load.",
        )
