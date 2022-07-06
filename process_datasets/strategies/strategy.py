from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

from process_datasets.utils.general import dict2list, list2dict


class Strategy(ABC):
    r"""Given a stream of input text, creates pretraining examples."""

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

    def __call__(self, batch: Dict[Any, List]) -> Dict[Any, List]:
        r""" Receive a batch of documents and return processed version.
        """
        batch = dict2list(batch)
        batch = self.process_batch(batch)
        batch = list2dict(batch)
        return batch

    @abstractmethod
    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        r""" Process a list of batches. """

    @staticmethod
    def add_arguments_to_argparse(parser: ArgumentParser):
        r""" Add strategy specific parameters to the cmd argument parser. """
        parser.add_argument(
            '--field', type=str, required=False, default='text', help="Field names in the dataset to consider."
        )
