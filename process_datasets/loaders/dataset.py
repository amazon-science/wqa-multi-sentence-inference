import logging
from argparse import ArgumentParser, Namespace

import datasets

from process_datasets.loaders.loader import Loader


class DatasetLoader(Loader):
    r""" Load a dataset from the datasets library. """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)

        if len(hparams.name) > 1:
            hparams.name, hparams.config = hparams.name
        else:
            hparams.config = None
            hparams.name = hparams.name[0]

        logging.info(f"Loading input dataset {hparams.name} with config {hparams.config}")
        dataset = datasets.load_dataset(hparams.name, hparams.config, keep_in_memory=hparams.keep_in_memory)
        self.dataset = dataset if hparams.split is None else dataset[hparams.split]

    def add_loader_specific_args(parser: ArgumentParser):
        super(DatasetLoader, DatasetLoader).add_loader_specific_args(parser)
        parser.add_argument('--name', type=str, required=True, nargs='+')
