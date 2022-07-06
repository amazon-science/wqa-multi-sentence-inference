from abc import ABC, abstractmethod
from argparse import ArgumentParser

from pytorch_lightning.utilities import rank_zero_warn
from transformers import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers_lightning.models import TransformersModel


class BaseModel(TransformersModel, ABC):

    config_class: PretrainedConfig
    model_class: PreTrainedModel
    tokenizer_class: PreTrainedTokenizer

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.save_hyperparameters(hyperparameters)

        if self.hyperparameters.pre_trained_config is None:
            self.hyperparameters.pre_trained_config = self.hyperparameters.pre_trained_model
            rank_zero_warn('Found None `pre_trained_config`, setting equal to `pre_trained_model`')

        if self.hyperparameters.pre_trained_tokenizer is None:
            self.hyperparameters.pre_trained_tokenizer = self.hyperparameters.pre_trained_model
            rank_zero_warn('Found None `pre_trained_tokenizer`, setting equal to `pre_trained_model`')

        assert self.hyperparameters.pre_trained_config is not None, (
            "Cannot instantiate model without a pre_trained_config."
        )

        self.config = self.setup_config()
        self.model = self.setup_model(self.config)
        self.tokenizer = self.setup_tokenizer()

    @abstractmethod
    def setup_config(self) -> PretrainedConfig:
        r""" Return the config instance. """

    @abstractmethod
    def setup_model(self, config: PretrainedConfig) -> PreTrainedModel:
        r""" Return the model instance. """

    @abstractmethod
    def setup_tokenizer(self) -> PreTrainedTokenizer:
        r""" Return the tokenizer instance. """

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser):
        super(BaseModel, BaseModel).add_model_specific_args(parser)
        # add pre_trained model, tokenizer and config arguments. default config and tokenizer to model if missing
        parser.add_argument('--pre_trained_model', type=str, required=False, default=None)
        parser.add_argument('--pre_trained_tokenizer', type=str, required=False, default=None)
        parser.add_argument('--pre_trained_config', type=str, required=False, default=None)
        parser.add_argument('--num_labels', type=int, required=False, default=2)
