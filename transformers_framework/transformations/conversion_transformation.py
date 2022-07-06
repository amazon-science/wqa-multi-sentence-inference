from argparse import Namespace
from types import GeneratorType
from typing import Dict, Generator, Union

from transformers_framework.transformations.transformation import Transformation
from transformers_framework.utilities.functional import apply_to_generator
from transformers_framework.utilities.structures import JointwiseSample, PairwiseSample


# Dict -> PairwiseSample
class DictSample2PairwiseSampleTransformation(Transformation):
    r""" Transforms DictSample to DataSample. """

    def __init__(
        self,
        hyperparameters: Namespace,
        first_field: str,
        second_field: str,
        key_field: str = 'key',
        label_field: str = 'label',
        score_field: str = 'score',
    ):
        super().__init__(hyperparameters)
        self.first_field = first_field
        self.second_field = second_field
        self.key_field = key_field
        self.label_field = label_field
        self.score_field = score_field

    def dict_sample_to_pairwise_sample(self, sample: Dict) -> PairwiseSample:
        r""" Transforming Dict instances to PairwiseSample. """
        assert isinstance(sample, Dict), f"input must be of type Dict, found {sample.__class__.__name__}"
        return PairwiseSample(
            first=" ".join(sample[f] for f in self.first_field.split(":")),
            second=" ".join(sample[f] for f in self.second_field.split(":")),
            key=sample[self.key_field] if self.key_field in sample else None,
            label=sample[self.label_field] if self.label_field in sample else None,
            score=sample[self.score_field] if self.score_field in sample else None,
        )

    def __call__(
        self,
        samples: Union[Generator[Dict, None, None], Dict]
    ) -> Union[Generator[PairwiseSample, None, None], PairwiseSample]:
        r""" Transforming Dict instances to PairwiseSample. """
        if isinstance(samples, GeneratorType):
            return apply_to_generator(samples, self.dict_sample_to_pairwise_sample)
        else:
            return self.dict_sample_to_pairwise_sample(samples)


# Dict -> JointwiseSample
class DictSample2JointwiseSampleTransformation(Transformation):
    r""" Transforms DictSample to JointwiseSample. """

    def __init__(
        self,
        hyperparameters: Namespace,
        first_field: str,
        seconds_field: str,
        key_field: str = 'key',
        label_field: str = 'label',
        score_field: str = 'score',
    ):
        super().__init__(hyperparameters)

        assert ":" not in first_field
        assert ":" not in seconds_field

        self.first_field = first_field
        self.seconds_field = seconds_field
        self.key_field = key_field
        self.label_field = label_field
        self.score_field = score_field

    def dict_sample_to_jointwise_sample(self, sample: Dict) -> JointwiseSample:
        r""" Transform a Dict instance to JointwiseSample. """
        assert isinstance(sample, Dict), f"input must be of type Dict, found {sample.__class__.__name__}"
        return JointwiseSample(
            first=sample[self.first_field],
            seconds=sample[self.seconds_field],
            key=sample[self.key_field] if self.key_field in sample else None,
            label=sample[self.label_field] if self.label_field in sample else None,
            score=sample[self.score_field] if self.score_field in sample else None,
            valid=[True] * len(sample[self.seconds_field]),
        )

    def __call__(
        self,
        samples: Union[Generator[Dict, None, None], Dict]
    ) -> Union[Generator[JointwiseSample, None, None], JointwiseSample]:
        r""" Transform a Dict instance to JointwiseSample. """
        if isinstance(samples, GeneratorType):
            return apply_to_generator(samples, self.dict_sample_to_jointwise_sample)
        else:
            return self.dict_sample_to_jointwise_sample(samples)
