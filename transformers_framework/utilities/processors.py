from typing import Union

from transformers import PreTrainedTokenizer

from transformers_framework.utilities.structures import DataSample, JointwiseSample, PairwiseSample
from transformers_framework.utilities.tokenization import encode_many_sequences, encode_pair, encode_sequences


def data_processor(
    sample: DataSample,
    tokenizer: PreTrainedTokenizer,
    max_sequence_length: int,
    chain: bool,
):
    r""" Generinc encoder. Encodes data by concatenating on the msl axis. """
    encoded = encode_sequences(sample.data, tokenizer, max_sequence_length, chain=chain)

    if sample.label is not None:
        encoded['labels'] = sample.label
    if sample.key is not None:
        encoded['keys'] = sample.key

    return encoded


def pair_processor(
    sample: PairwiseSample,
    tokenizer: PreTrainedTokenizer,
    separated: bool,
    max_sequence_length: int,
    chain: bool,
    truncation: Union[int, str] = True,
    padding: str = "max_length",
    return_overflowing_tokens: bool = False,
    return_offsets_mapping: bool = False,
    stride: int = 0,
    allow_null_second: bool = False,
):
    r""" Encode a pair for pairwise training. """
    # TODO: remove comment
    # assert sample.first is not None and (sample.second is not None or allow_null_second is True)

    if separated:
        encoded = encode_sequences([sample.first, sample.second], tokenizer, max_sequence_length, chain=chain)
    else:
        encoded = encode_pair(
            sample.first,
            sample.second,
            tokenizer,
            max_sequence_length,
            truncation=truncation,
            padding=padding,
            return_overflowing_tokens=return_overflowing_tokens,
            return_offsets_mapping=return_offsets_mapping,
            stride=stride,
            allow_null_second=allow_null_second,
        )

    if sample.label is not None:
        encoded['labels'] = sample.label
    if sample.key is not None:
        encoded['keys'] = sample.key

    return encoded


def joint_processor(
    sample: JointwiseSample,
    tokenizer: PreTrainedTokenizer,
    separated: bool,
    max_sequence_length: int,
    reduce_labels: bool,
):
    r""" Encoding for Jointwise models.
    Encodes a sequence on sentences with internal padding by concatenating along the msl axis.
    """
    if separated:
        encoded = encode_sequences([sample.first] + sample.seconds, tokenizer, max_sequence_length, chain=True)
    else:
        encoded = encode_many_sequences([sample.first] + sample.seconds, tokenizer, max_sequence_length)

    if sample.label is not None:
        encoded['labels'] = sample.label[0] if reduce_labels else sample.label
    if sample.key is not None:
        encoded['keys'] = sample.key
    if sample.valid is not None:
        encoded['valid'] = sample.valid

    return encoded
