import random
from argparse import Namespace
from types import GeneratorType
from typing import Generator, Sequence, Union
from collections import Iterable

from transformers_lightning.language_modeling import IGNORE_IDX

from transformers_framework.transformations.transformation import Transformation
from transformers_framework.utilities.functional import (
    apply_to_generator,
    none_if_all_none,
    pad_sequence,
    special_zip,
    split,
)
from transformers_framework.utilities.structures import JointwiseSample


selection_possibilities = ("random", "best", "worst", "all")


class FilterJointSampleOnKTransformation(Transformation):
    r""" Filter JointwiseSample instances based on `k` and `selection` hyperparameters.
    Eventually pad to match `k` length.
    """

    def __init__(
        self,
        hyperparameters: Namespace,
        padding: bool = False,
        seed: int = None,
    ):
        super().__init__(hyperparameters)

        assert isinstance(padding, bool), "padding argument must be boolean"

        self.padding = padding
        self.pad_score = float('-inf')
        self.pad_string = ""
        self.pad_label = IGNORE_IDX
        self.pad_valid = False

        assert 'shuffle_candidates' in self.hyperparameters and isinstance(
            self.hyperparameters.shuffle_candidates, bool
        )
        assert 'k' in self.hyperparameters and (
            self.hyperparameters.k is None or isinstance(self.hyperparameters.k, int)
        )
        assert 'selection' in self.hyperparameters and self.hyperparameters.selection in selection_possibilities
        self.requires_scores = (
            'selection' in self.hyperparameters and self.hyperparameters.selection in ('best', 'worst')
        )

        # random generator provided or default
        self.random_generator = random.Random(seed) if seed is not None else random

    def filter_on_k(self, sample: JointwiseSample) -> Union[Generator[JointwiseSample, None, None], JointwiseSample]:
        r""" Filter example on the number of candidates. If selection is all, the number of output
        examples may be greater than the inputs. """

        # some tasks have just a single label, like fact checking
        single_label = not isinstance(sample.label, Iterable)
        if single_label:
            sample.label = [sample.label] * len(sample.seconds)

        # zipping everything together to apply always the same modifications
        score_label_candidates_valid = list(special_zip(sample.score, sample.label, sample.seconds, sample.valid))

        # take best / worst / random k entries from data
        if self.hyperparameters.k is not None:

            # take all elements in batches of k
            if self.hyperparameters.selection == "all":
                # shuffling here is needed to shuffle also among different subsets
                if self.hyperparameters.shuffle_candidates is True:
                    self.random_generator.shuffle(score_label_candidates_valid)

                yield from split(score_label_candidates_valid, self.hyperparameters.k, drop_last=False)

            # randomly select k candidates if there are more than k
            elif self.hyperparameters.selection == "random":
                if len(score_label_candidates_valid) > self.hyperparameters.k:
                    score_label_candidates_valid = self.random_generator.sample(
                        score_label_candidates_valid, k=self.hyperparameters.k
                    )
                yield score_label_candidates_valid

            else:
                # sort based on score
                score_label_candidates_valid = sorted(score_label_candidates_valid)

                # take the best k elements based on the score
                if self.hyperparameters.selection == "best":
                    yield score_label_candidates_valid[-self.hyperparameters.k:]

                # take the worst k elements based on the score
                elif self.hyperparameters.selection == "worst":
                    yield score_label_candidates_valid[:self.hyperparameters.k]

                else:
                    raise ValueError(f"selection `{self.hyperparameters.selection}` not valid")

        # append everything since not filtering on k is performed
        else:
            yield score_label_candidates_valid

    def pad_to_k(self, key: int, first: str, score_label_candidates_valid: Sequence) -> JointwiseSample:
        r""" Eventually pad to length k for every seconds, label and score, valid. """
        # shuffle, necessary since best / worst may have sorted everything
        if self.hyperparameters.shuffle_candidates is True:
            self.random_generator.shuffle(score_label_candidates_valid)

        # extract original arrays
        score, label, seconds, valid = zip(*score_label_candidates_valid)

        # create new sample otherwise many samples will be linked to the same one
        new_sample = JointwiseSample(
            key=key,
            first=first,
            seconds=list(seconds),
            label=list(label),
            score=list(score),
            valid=list(valid),
        )

        # if are all none, put none e basta
        new_sample.score = none_if_all_none(new_sample.score)
        new_sample.label = none_if_all_none(new_sample.label)

        # eventually pad on k
        if self.padding is True:
            new_sample.seconds = pad_sequence(new_sample.seconds, self.pad_string, self.hyperparameters.k)
            new_sample.valid = pad_sequence(new_sample.valid, self.pad_valid, self.hyperparameters.k)

            if new_sample.label is not None:
                new_sample.label = pad_sequence(new_sample.label, self.pad_label, self.hyperparameters.k)

            if new_sample.score is not None:
                new_sample.score = pad_sequence(new_sample.score, self.pad_score, self.hyperparameters.k)

        return new_sample

    def _filter_and_pad(
        self,
        sample: JointwiseSample,
    ) -> Union[Generator[JointwiseSample, None, None], JointwiseSample]:
        r""" Filter and pad, managing singles and generators.
        """
        assert not self.requires_scores or sample.score is not None, (
            "score required to `selection` on best or worst `seconds`"
        )

        for res in self.filter_on_k(sample):
            yield self.pad_to_k(sample.key, sample.first, res)

    def filter_and_pad(
        self,
        samples: Union[Generator[JointwiseSample, None, None], JointwiseSample],
    ) -> Union[Generator[JointwiseSample, None, None], JointwiseSample]:
        r""" Filter and pad, managing singles and generators.
        """
        if isinstance(samples, GeneratorType):
            for sample in samples:
                yield from self._filter_and_pad(sample)
        else:
            yield from self._filter_and_pad(samples)

    def __call__(
        self,
        samples: Union[Generator[JointwiseSample, None, None], JointwiseSample]
    ) -> Union[Generator[JointwiseSample, None, None], JointwiseSample]:
        r""" It may return many JointwiseSample for a single JointwiseSample if selection is `all`. """
        if isinstance(samples, GeneratorType):
            return self.filter_and_pad(samples)
        else:
            if self.hyperparameters.selection != 'all':
                samples = list(self.filter_and_pad(samples))
                assert len(samples) == 1
                return samples[0]
            else:
                return self.filter_and_pad(samples)


class FilterJointwiseSampleOnScoreAndLabelTransformation(Transformation):
    r""" Filter JointwiseSample instances based on `k` and `selection` hyperparameters.
    Eventually pad to match `k` length.
    """

    def __init__(self, hyperparameters: Namespace):
        super().__init__(hyperparameters)

        assert 'min_threshold' in self.hyperparameters and (
            self.hyperparameters.min_threshold is None or isinstance(self.hyperparameters.min_threshold, float)
        )
        assert 'max_threshold' in self.hyperparameters and (
            self.hyperparameters.max_threshold is None or isinstance(self.hyperparameters.max_threshold, float)
        )
        assert 'max_positives' in self.hyperparameters and (
            self.hyperparameters.max_positives is None or isinstance(self.hyperparameters.max_positives, int)
        )
        assert 'max_negatives' in self.hyperparameters and (
            self.hyperparameters.max_negatives is None or isinstance(self.hyperparameters.max_negatives, int)
        )

        self.requires_scores = (
            (self.hyperparameters.min_threshold is not None) or (self.hyperparameters.max_threshold is not None)
        )
        self.requires_labels = (
            (self.hyperparameters.max_positives is not None) or (self.hyperparameters.max_negatives is not None)
        )

    def __call__(
        self,
        samples: Union[Generator[JointwiseSample, None, None], JointwiseSample]
    ) -> Union[Generator[JointwiseSample, None, None], JointwiseSample]:
        r""" Filter candidates of every example based on number of negative/positive labels and scores thresholds.
        """
        if isinstance(samples, GeneratorType):
            return apply_to_generator(samples, self.filter_on_score_and_label)
        else:
            return self.filter_on_score_and_label(samples)

    def filter_on_score_and_label(self, sample: JointwiseSample) -> Generator[JointwiseSample, None, None]:
        r""" Filter every multi-pair input sample on score with thresholds and number of positive/negatives.
        """
        # filter on thresholds or labels
        assert not self.requires_scores or sample.score is not None
        assert not self.requires_labels or sample.label is not None

        # some tasks have just a single label, like fact checking
        single_label = not isinstance(sample.label, Iterable)
        if single_label:
            sample.label = [sample.label] * len(sample.seconds)

        # zipping everything together to apply always the same modifications
        score_label_candidates_valid = list(special_zip(sample.score, sample.label, sample.seconds, sample.valid))

        # filter candidates with a low score
        if self.hyperparameters.min_threshold is not None:
            score_label_candidates_valid = [
                (score, label, candidate) for score, label, candidate in score_label_candidates_valid
                if score >= self.hyperparameters.min_threshold
            ]

        # filter candidates with a high score
        if self.hyperparameters.max_threshold is not None:
            score_label_candidates_valid = [
                (score, label, candidate) for score, label, candidate in score_label_candidates_valid
                if score <= self.hyperparameters.max_threshold
            ]

        # filter candidates with a positive label
        if self.hyperparameters.max_positives is not None:
            # get position of positives
            positive_position = [i for i, (_, label, _) in enumerate(score_label_candidates_valid) if label == 1]

            if len(positive_position) > self.hyperparameters.max_positives:
                # subsample positives
                positive_position = random.sample(positive_position, k=self.hyperparameters.max_positives)
                # filter
                score_label_candidates_valid = [
                    (score, label, candidate)
                    for i, (score, label, candidate) in enumerate(score_label_candidates_valid)
                    if (label == 0) or (i in positive_position)
                ]

        # filter candidates with a negative label
        if self.hyperparameters.max_negatives is not None:
            # get position of negatives
            negative_position = [i for i, (_, label, _) in enumerate(score_label_candidates_valid) if label == 0]

            if len(negative_position) > self.hyperparameters.max_negatives:
                # subsample negatives
                negative_position = random.sample(negative_position, k=self.hyperparameters.max_negatives)
                # filter
                score_label_candidates_valid = [
                    (score, label, candidate)
                    for i, (score, label, candidate) in enumerate(score_label_candidates_valid)
                    if (label == 1) or (i in negative_position)
                ]

        sample.score, sample.label, sample.seconds, sample.valid = zip(*score_label_candidates_valid)
        sample.score = none_if_all_none(sample.score)
        sample.label = none_if_all_none(sample.label)

        return sample
