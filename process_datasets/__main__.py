import logging
import random
from argparse import ArgumentParser

from datasets import DatasetDict
from psutil import cpu_count
from transformers_lightning.utils import get_classes_from_module

from process_datasets import loaders, strategies
from process_datasets.loaders.loader import Loader
from process_datasets.strategies.strategy import Strategy


ALL_STRATEGIES = get_classes_from_module(strategies, parent=Strategy)
ALL_LOADERS = get_classes_from_module(loaders, parent=Loader)


def main(hparams):

    logging.info("Setting seed...")
    random.seed(hparams.seed)

    # Create instances
    logging.info("Creating loader and strategy instances...")
    loader = ALL_LOADERS[hparams.loader](hparams)
    strategy = ALL_STRATEGIES[hparams.strategy](hparams)

    logging.info("Starting pipeline...")

    # Data loading
    dataset = loader()

    # Data processing
    kwargs = dict(num_proc=hparams.num_proc) if hparams.num_proc > 0 else dict()
    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict(**{
            k: dataset[k].map(
                strategy,
                batched=True,
                batch_size=hparams.batch_size,
                remove_columns=dataset[k].column_names,
                keep_in_memory=False,
                **kwargs,
            ) for k in dataset
        })
    else:
        dataset = dataset.map(
            strategy,
            batched=True,
            batch_size=hparams.batch_size,
            remove_columns=dataset.column_names,
            keep_in_memory=False,
            **kwargs,
        )

    # Data writing
    logging.info("Writing to disk")
    dataset.save_to_disk(hparams.output_folder)

    logging.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(f"Create pretraining datasets")
    parser.add_argument(
        '--loader', type=str, required=True, choices=ALL_LOADERS, help="Loader class to load data"
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=False,
        default='Sentence2SentenceStrategy',
        choices=ALL_STRATEGIES.keys(),
        help="Strategy to use to create the dataset",
    )
    parser.add_argument('--output_folder', type=str, required=True, help="Output folder")
    parser.add_argument('--seed', default=1337, required=False, type=int, help="Seed for reproducibility.")
    parser.add_argument(
        '--batch_size', default=10000, type=int, required=False, help="How many input examples to process together."
    )
    parser.add_argument('--num_proc', type=int, default=cpu_count(), required=False, help="How many process to use.")
    # add strategy parameters
    tmp_hparams, _ = parser.parse_known_args()

    loader_class = ALL_LOADERS[tmp_hparams.loader]
    strategy_class = ALL_STRATEGIES[tmp_hparams.strategy]

    loader_class.add_loader_specific_args(parser)
    strategy_class.add_arguments_to_argparse(parser)

    hparams = parser.parse_args()
    main(hparams)
