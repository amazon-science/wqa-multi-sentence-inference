import logging
import os
from argparse import ArgumentParser, Namespace

import datasets

from process_datasets.loaders.loader import Loader


class DiskLoader(Loader):
    r""" Load a dataset from disk. """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        assert os.path.isdir(hparams.input_folder), "Input folder does not exist"

        logging.info(f"Loading input dataset from disk")
        dataset = datasets.load_from_disk(hparams.input_folder, keep_in_memory=hparams.keep_in_memory)
        self.dataset = dataset if hparams.split is None else dataset[hparams.split]

    def add_loader_specific_args(parser: ArgumentParser):
        super(DiskLoader, DiskLoader).add_loader_specific_args(parser)
        parser.add_argument('--input_folder', type=str, required=True)
