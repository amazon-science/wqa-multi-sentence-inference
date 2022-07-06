from argparse import ArgumentParser, Namespace
from typing import Dict

from pytorch_lightning.trainer.states import TrainerFn
from transformers import PreTrainedTokenizer

from transformers_framework.adapters.map_adapters.arrow.pairwise_adapter import PairwiseArrowAdapter
from transformers_framework.transformations.conversion_transformation import DictSample2JointwiseSampleTransformation
from transformers_framework.transformations.filtering_transformation import (
    FilterJointSampleOnKTransformation,
    FilterJointwiseSampleOnScoreAndLabelTransformation,
    selection_possibilities,
)
from transformers_framework.transformations.transformation import Transformation, TransformationsConcatenation
from transformers_framework.utilities import JointwiseSample
from transformers_framework.utilities.processors import joint_processor


class JointwiseArrowAdapter(PairwiseArrowAdapter):
    r""" Jointwise version of Arrow File readers, which implements filtering on scores and limits. """

    def __init__(
        self,
        hyperparameters: Namespace,
        tokenizer: PreTrainedTokenizer,
        stage: TrainerFn,
        seed: int = 0,
    ) -> None:
        super().__init__(hyperparameters, tokenizer, stage, seed=seed)

        # arguments check
        assert type(self) != JointwiseArrowAdapter or isinstance(hyperparameters.k, int), (
            f"provided `k` {hyperparameters.k} is not an integer"
        )
        assert type(self) != JointwiseArrowAdapter or (
            hyperparameters.selection is not None and hyperparameters.selection in selection_possibilities
        ), (
            f"provided `selection` {hyperparameters.selection} is not among "
            f"the accepted values {selection_possibilities} or is `None`"
        )
        assert type(self) != JointwiseArrowAdapter or (
            hyperparameters.selection != 'all' or hyperparameters.force_load_dataset_in_memory is True
        ), "selection=`all` requires `force_load_dataset_in_memory` to be true"
        assert hyperparameters.min_threshold is None or isinstance(hyperparameters.min_threshold, float), (
            "`min_threshold` must be a float or None"
        )
        assert hyperparameters.max_threshold is None or isinstance(hyperparameters.max_threshold, float), (
            "`max_threshold` must be a float or None"
        )
        assert hyperparameters.max_positives is None or isinstance(hyperparameters.max_positives, int), (
            "`max_positives` must be a int or None"
        )
        assert hyperparameters.max_negatives is None or isinstance(hyperparameters.max_negatives, int), (
            "`max_negatives` must be a int or None"
        )
        assert type(self) != JointwiseArrowAdapter or hyperparameters.shuffle_candidates is False or (
            isinstance(hyperparameters.reload_dataloaders_every_n_epochs, int)
            and hyperparameters.reload_dataloaders_every_n_epochs > 0
        ), "`shuffle_candidates` requires not None `reload_dataloaders_every_n_epochs` greater than 0"
        assert type(self) != JointwiseArrowAdapter or hyperparameters.separated is True, (
            "`separated` must be True"
        )
        assert type(self) != JointwiseArrowAdapter or len(self.hyperparameters.field_names) == 2, (
            "`field_names` must have length 2"
        )
        assert type(self) != JointwiseArrowAdapter or all(
            ":" not in field for field in self.hyperparameters.field_names
        ), "every `field_names` must not contain the `:`"

    def __iter__(self):
        for idx, sample in enumerate(self.data):
            self.check(sample, idx)  # check sample is valid and raise warning
            if self.hyperparameters.selection == 'all':
                yield from self.transformations(sample)
            else:
                yield self.transformations(sample)

    def __get_transformations__(self) -> Transformation:
        return TransformationsConcatenation(
            self.hyperparameters,
            DictSample2JointwiseSampleTransformation(
                self.hyperparameters,
                first_field=self.hyperparameters.field_names[0],
                seconds_field=self.hyperparameters.field_names[1],
                key_field=self.hyperparameters.key_name,
                label_field=self.hyperparameters.label_name,
                score_field=self.hyperparameters.score_name,
            ),  # Dict -> JointwiseSample
            FilterJointwiseSampleOnScoreAndLabelTransformation(self.hyperparameters),  # Filter on score and labels
            FilterJointSampleOnKTransformation(  # Filter on K
                self.hyperparameters,
                padding=True,
                seed=self.seed,
            )  # -> JointwiseSample
        )

    def preprocess_line(self, sample: JointwiseSample) -> Dict:
        r"""
        Process a line. The structure of each line is exactly
        the same returned by the __iter__ method. Here you should do data preparation
        for the actual model being trained. This is a good place to do batch tokenization,
        padding and so on.
        """
        return joint_processor(
            sample,
            tokenizer=self.tokenizer,
            separated=self.hyperparameters.separated,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            reduce_labels=self.hyperparameters.reduce_labels,
        )

    @staticmethod
    def add_adapter_specific_args(parser: ArgumentParser):
        super(JointwiseArrowAdapter, JointwiseArrowAdapter).add_adapter_specific_args(parser)
        parser.add_argument(
            '--shuffle_candidates', action="store_true", help="Shuffle candidates when using `k` to select them"
        )
        parser.add_argument(
            '--min_threshold',
            type=float,
            required=False,
            default=None,
            help="Lower threshold for candidates filtering",
        )
        parser.add_argument(
            '--max_threshold',
            type=float,
            required=False,
            default=None,
            help="Upper threshold for candidates filtering",
        )
        parser.add_argument(
            '--max_negatives', type=int, required=False, default=None, help="Lower threshold for candidates filtering"
        )
        parser.add_argument(
            '--max_positives', type=int, required=False, default=None, help="Upper threshold for candidates filtering"
        )
        parser.add_argument(
            '-k', type=int, required=False, default=None, help="Number of candidates to be cumulated on a row"
        )
        parser.add_argument(
            '--selection',
            type=str,
            required=False,
            default=None,
            choices=selection_possibilities,
            help="How to get candidates if `-k` is defined",
        )
        parser.add_argument(
            '--reduce_labels',
            action="store_true",
            help="Whether to reduce label to single dim when using k. Labels must be all equal in the same group",
        )
        parser.add_argument(
            '--extended_token_type_ids',
            type=int,
            default=None,
            help="How many extended TT ids should be generated.",
        )
