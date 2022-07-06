from abc import abstractmethod
from argparse import ArgumentParser, Namespace

from transformers import PreTrainedTokenizer
from transformers_lightning.adapters import SuperAdapter

from transformers_framework.transformations.transformation import Transformation, TransformationsConcatenation


class TransformersAdapter(SuperAdapter):

    def __init__(
        self,
        hyperparameters: Namespace,
        tokenizer: PreTrainedTokenizer,
        stage_name: str,
        seed: int = 0,
    ) -> None:
        super().__init__(hyperparameters)

        self.tokenizer = tokenizer
        self.stage_name = stage_name
        self.seed = seed
        self.transformations = self.__get_transformations__()

    def __get_transformations__(self) -> Transformation:
        return TransformationsConcatenation(self.hyperparameters)  # empty transformations

    @abstractmethod
    def is_active(self) -> bool:
        r""" Return True or False based on whether this adapter could return data or not. """

    @staticmethod
    def add_adapter_instance_specific_args(parser: ArgumentParser, stage_name: str):
        r""" In the case many adapters are used, it could be useful
        to organize the arguments of every adapter using a different prefix.
        Put here all the arguments that are not shared by every adapter instance, for
        example the path of the data on the disk. """
