from transformers import RobertaTokenizerFast

from transformers_framework.architectures.roberta.modeling_config import RobertaJointConfig
from transformers_framework.architectures.roberta.modeling_joint_roberta import JointRobertaForSequenceClassification
from transformers_framework.models.joint.fact_checking.base import BaseJointFactChecking


class RobertaJointFactChecking(BaseJointFactChecking):

    def setup_config(self) -> RobertaJointConfig:
        return RobertaJointConfig.from_pretrained(
            self.hyperparameters.pre_trained_config,
            k=self.hyperparameters.k,
            sentence_msl=self.hyperparameters.max_sequence_length,
            num_labels=self.hyperparameters.num_labels,
            head_type=self.hyperparameters.head_type,
        )

    def setup_model(self, config: RobertaJointConfig) -> JointRobertaForSequenceClassification:
        if self.hyperparameters.pre_trained_model is None:
            return JointRobertaForSequenceClassification(config)
        else:
            return JointRobertaForSequenceClassification.from_pretrained(
                self.hyperparameters.pre_trained_model, config=config, ignore_mismatched_sizes=True
            )

    def setup_tokenizer(self) -> RobertaTokenizerFast:
        return RobertaTokenizerFast.from_pretrained(self.hyperparameters.pre_trained_tokenizer)
