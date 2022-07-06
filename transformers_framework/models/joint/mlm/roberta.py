from transformers import RobertaTokenizerFast

from transformers_framework.architectures.roberta.modeling_config import RobertaJointConfig
from transformers_framework.architectures.roberta.modeling_joint_roberta import JointRobertaForMaskedLM
from transformers_framework.models.joint.mlm.base import BaseJointMLM


class RobertaJointMLM(BaseJointMLM):

    def setup_config(self) -> RobertaJointConfig:
        return RobertaJointConfig.from_pretrained(
            self.hyperparameters.pre_trained_config,
            k=self.hyperparameters.k,
            sentence_msl=self.hyperparameters.max_sequence_length,
        )

    def setup_model(self, config: RobertaJointConfig) -> JointRobertaForMaskedLM:
        if self.hyperparameters.pre_trained_model is None:
            return JointRobertaForMaskedLM(config)
        else:
            return JointRobertaForMaskedLM.from_pretrained(
                self.hyperparameters.pre_trained_model, config=config, ignore_mismatched_sizes=True
            )

    def setup_tokenizer(self) -> RobertaTokenizerFast:
        return RobertaTokenizerFast.from_pretrained(self.hyperparameters.pre_trained_tokenizer)
