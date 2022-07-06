from itertools import chain as iterchain
from typing import Dict, Sequence, Union

from transformers import PreTrainedTokenizer

from transformers_framework.architectures.modeling_tokenizer import ExtendedTokenizerFast


def encode_sequences(
    sequences: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    max_sequence_length: int,
    chain: bool = True,
    extended_token_type_ids: int = None,
) -> Dict:
    r""" Encode a sequence of sentences as
    [CLS] question [SEP] [CLS] candidate_1 [SEP] [CLS] candidate_2 [SEP] ...
    or
    <s> question </s> <s> candidate_1 </s> <s> candidate_2 </s> ...
    such that the total length is equal to `max_sequence_length` * len(sequences)

    If chain, chain every sequence before returning.
    """
    encoded = tokenizer(
        sequences,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        max_length=max_sequence_length,
        truncation=True,
        padding='max_length',
    )

    if extended_token_type_ids is not None:
        encoded['token_type_ids'] = [
            [min(i, extended_token_type_ids)] * len(ids)
            for i, ids in enumerate(encoded['input_ids'])
        ]

    if chain:
        encoded = {k: list(iterchain(*v)) for k, v in encoded.items()}

    return encoded


def encode_many_sequences(
    sequences: Sequence[str],
    tokenizer: ExtendedTokenizerFast,
    max_sequence_length: int,
    extended_token_type_ids: int = None,
) -> Dict:
    r""" Encode a list of sequences as
    [CLS] first [SEP] second1 [SEP] second2 [SEP] ... [SEP]
    or
    <s> first </s></s> candidate1 </s></s> candidate2 </s></s> ... </s></s>
    such that the total length is equal to `max_sequence_length`.
    """
    assert isinstance(tokenizer, ExtendedTokenizerFast), (
        "Cannot use `encode_many_sequences` without ExtendedTokenizer"
    )

    encoded = tokenizer.encode_many(
        sequences,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        max_length=max_sequence_length,
        truncation='longest_first',
        padding="max_length",
        extended_token_type_ids=extended_token_type_ids,
    )
    return encoded


def encode_pair(
    first: str,
    second: str,
    tokenizer: PreTrainedTokenizer,
    max_sequence_length: int,
    truncation: Union[int, str] = True,
    padding: str = "max_length",
    return_overflowing_tokens: bool = False,
    return_offsets_mapping: bool = False,
    stride: int = 0,
    allow_null_second: bool = False,
) -> Dict:
    r""" Encode a first-second pair as
    [CLS] first [SEP] second [SEP]
    or
    <s> first </s></s> second </s>
    such that the total length is equal to `max_sequence_length`.
    """
    tok_args = dict(
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        max_length=max_sequence_length,
        truncation=truncation,
        padding=padding,
        return_overflowing_tokens=return_overflowing_tokens,
        return_offsets_mapping=return_offsets_mapping,
        stride=stride,
    )

    if second is None and allow_null_second is True:
        encoded = tokenizer(first, **tok_args)
    else:
        encoded = tokenizer(first, second, **tok_args)
    return encoded
