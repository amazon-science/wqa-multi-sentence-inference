from argparse import ArgumentParser

from torchmetrics import Accuracy
from transformers_lightning.language_modeling import MaskedLanguageModeling

from transformers_framework.models.base.base import BaseModel


class BaseModelMLM(BaseModel):

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)

        self.mlm = MaskedLanguageModeling(
            self.tokenizer,
            probability=hyperparameters.probability,
            probability_masked=hyperparameters.probability_masked,
            probability_replaced=hyperparameters.probability_replaced,
        )

        self.train_mlm_acc = Accuracy()
        self.valid_mlm_acc = Accuracy()
        self.test_mlm_acc = Accuracy()

    def add_model_specific_args(parser: ArgumentParser):
        super(BaseModelMLM, BaseModelMLM).add_model_specific_args(parser)
        # mlm specific arguments
        parser.add_argument('--probability', type=float, default=0.15)
        parser.add_argument('--probability_masked', type=float, default=0.80)
        parser.add_argument('--probability_replaced', type=float, default=0.10)
