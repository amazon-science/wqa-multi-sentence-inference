import logging
import os
from argparse import ArgumentParser, Namespace

from datasets import load_dataset

from process_datasets.loaders.loader import Loader


class JsonLoader(Loader):
    r""" Load a dataset from a bunch of JSONL files. """

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        assert os.path.isdir(hparams.input_folder), "Input folder does not exist"

        logging.info(f"Loading dataset from local json files in folder {hparams.input_folder}")
        filepaths = [
            os.path.join(hparams.input_folder, f)
            for f in os.listdir(hparams.input_folder)
            if f.endswith('.json') or f.endswith('.jsonl')
        ]
        dataset = load_dataset(
            'json', data_files=filepaths, keep_in_memory=hparams.keep_in_memory
        )
        self.dataset = dataset if hparams.split is None else dataset[hparams.split]

    def add_loader_specific_args(parser: ArgumentParser):
        super(JsonLoader, JsonLoader).add_loader_specific_args(parser)
        parser.add_argument('-i', '--input_folder', type=str, required=True, help="Input data folder.")
