from dataclasses import dataclass
from typing import Sequence, Union


@dataclass
class Sample:
    pass  # just a super class


@dataclass
class DataSample(Sample):
    data: Sequence[str]  # textual data inputs
    key: Union[int, None]  # an unique identifier of the input example
    label: Union[int, None]  # label as integer
    score: Union[float, None]  # score assigned to the textual example


@dataclass
class PairwiseSample(Sample):
    key: Union[int, None]  # an unique identifier of the input example
    first: str  # first sentence as string
    second: str  # second sentence as string
    label: Union[int, None]  # label as integer
    score: Union[float, None]  # score assigned to the textual example


@dataclass
class JointwiseSample(Sample):
    key: Union[int, None]  # an unique identifier of the input example
    first: str  # first sentence as string
    seconds: Sequence[str]  # second sentences as strings
    label: Union[Sequence[int], None]  # labels as integers
    score: Union[Sequence[float], None]  # scores assigned to the textual examples
    valid: Union[Sequence[float], None]  # valid positions (not padded)
