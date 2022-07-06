from argparse import ArgumentParser, Namespace
from typing import Dict

from pytorch_lightning.trainer.states import TrainerFn
from transformers import PreTrainedTokenizer

from transformers_framework.adapters.map_adapters.arrow.arrow_adapter import ArrowAdapter
from transformers_framework.transformations.conversion_transformation import DictSample2PairwiseSampleTransformation
from transformers_framework.transformations.transformation import Transformation, TransformationsConcatenation
from transformers_framework.utilities.processors import pair_processor
from transformers_framework.utilities.structures import PairwiseSample


class PairwiseArrowAdapter(ArrowAdapter):
    r""" Pairwise version of Arrow File readers, which implements filtering on scores and limits. """

    def __init__(
        self,
        hyperparameters: Namespace,
        tokenizer: PreTrainedTokenizer,
        stage: TrainerFn,
        seed: int = 0,
    ) -> None:
        super().__init__(hyperparameters, tokenizer, stage, seed=seed)

        # arguments checks
        assert not hyperparameters.chain or hyperparameters.separated, (
            "`chain` requires `separated`"
        )
        assert type(self) != PairwiseArrowAdapter or len(hyperparameters.field_names) == 2, (
            "`field_names` must have length 2"
        )
        assert type(self) != PairwiseArrowAdapter or (
            hyperparameters.separated is False or hyperparameters.allow_null_second is False
        ), "`allow_null_second` not allowed with `separated`"

    def __get_transformations__(self) -> Transformation:
        return TransformationsConcatenation(
            self.hyperparameters,
            DictSample2PairwiseSampleTransformation(
                self.hyperparameters,
                first_field=self.hyperparameters.field_names[0],
                second_field=self.hyperparameters.field_names[1],
                key_field=self.hyperparameters.key_name,
                label_field=self.hyperparameters.label_name,
                score_field=self.hyperparameters.score_name,
            ),  # Dict -> PairwiseSample
        )

    def preprocess_line(self, sample: PairwiseSample) -> Dict:
        r"""
        Process a line. The structure of each line is exactly
        the same returned by the __iter__ method. Here you should do data preparation
        for the actual model being trained. This is a good place to do batch tokenization,
        padding and so on.
        """
        return pair_processor(
            sample,
            tokenizer=self.tokenizer,
            separated=self.hyperparameters.separated,
            max_sequence_length=self.hyperparameters.max_sequence_length,
            chain=self.hyperparameters.chain,
            allow_null_second=self.hyperparameters.allow_null_second,
        )

    @staticmethod
    def add_adapter_specific_args(parser: ArgumentParser):
        super(PairwiseArrowAdapter, PairwiseArrowAdapter).add_adapter_specific_args(parser)
        parser.add_argument(
            '--separated', action="store_true", help="Candidates are separated between question and answer"
        )
        parser.add_argument(
            '--score_name',
            required=False,
            default=None,
            help="Name of the score field"
        )
        parser.add_argument('--allow_null_second', action="store_true", help='Allow second field to be None.')
