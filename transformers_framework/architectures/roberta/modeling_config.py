from transformers.models.roberta.configuration_roberta import RobertaConfig


MONOLITHIC_HEAD_TYPES = ("IE_1", "IE_k", "AE_1", "AE_k")


class RobertaJointConfig(RobertaConfig):
    r"""
    The :class:`~RobertaJointConfig` class directly inherits :class:`~transformers.RobertaConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Args:
        k (:obj:`int`, `optional`, defaults to 5):
            Number of candidates to consider for each query.
        sentence_msl (:obj:`int`, `optional`, defaults to 64):
            Max length of each query or candidate.
        head_type (:obj:`str`, `optional`, defaults to 'IE_1'):
            The classification head type.
    """

    def __init__(self, k: int = 5, sentence_msl: int = 64, head_type: str = "IE_1", **kwargs):
        super().__init__(is_decoder=False, **kwargs)
        assert head_type in MONOLITHIC_HEAD_TYPES
        self.k = k
        self.sentence_msl = sentence_msl
        self.head_type = head_type
