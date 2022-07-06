from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Generator, Sequence, Union

from transformers_framework.utilities.structures import Sample


class Transformation(ABC):
    r""" Transformation class encodes a transformation from samples to other samples. """

    def __init__(self, hyperparameters: Namespace):
        self.hyperparameters = hyperparameters

    @abstractmethod
    def __call__(
        self,
        samples: Union[Generator[Sample, None, None], Sample]
    ) -> Union[Generator[Sample, None, None], Sample]:
        r""" Apply here the trasformation on a single element or on all elements. """


class TransformationsConcatenation(Transformation):
    r""" Concatenate a sequence of transformations and return elements after parsing with every transformation
    in the same order provided in the __init__. """

    def __init__(self, hyperparameters: Namespace, *transformations: Sequence[Transformation]):
        super().__init__(hyperparameters)
        self.transformations = list(transformations)

    def __call__(
        self,
        samples: Union[Generator[Sample, None, None], Sample]
    ) -> Union[Generator[Sample, None, None], Sample]:
        for transformation in self.transformations:
            samples = transformation(samples)
        return samples

    def __str__(self) -> str:
        return (
            f"<TransformationsConcatenation object at {id(self)} containing: "
            f"{[t.__class__.__name__ for t in self.transformations]}>"
        )

    def append_transformation(self, transformation: Transformation):
        r""" Append transformation to the list. """
        self.transformations.append(transformation)
