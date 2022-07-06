from argparse import ArgumentParser, Namespace

from datasets import Dataset


class Loader:

    dataset: Dataset

    def __init__(self, hparams: Namespace):
        self.hparams = hparams

    def __call__(self) -> Dataset:
        r""" Return dataset for input data. """
        return self.dataset

    def add_loader_specific_args(parser: ArgumentParser):
        parser.add_argument('--keep_in_memory', action="store_true", help="Whether to keep in memory input dataset.")
        parser.add_argument('--split', default=None, required=False, type=str, help="Split to be loaded")
