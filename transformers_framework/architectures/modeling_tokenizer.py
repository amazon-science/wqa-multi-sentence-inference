from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from transformers.file_utils import PaddingStrategy, TensorType
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class ExtendedTokenizerFast(PreTrainedTokenizerFast, ABC):

    def encode_many(
        self,
        texts: Union[TextInput, PreTokenizedInput, EncodedInput],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        extended_token_type_ids: int = None,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        r"""
        Tokenize and prepare for the model many consecutive sequences.

        Args:
            texts (:obj:`str`, :obj:`List[str]` or `List[List[int]]`:
                The sequences to be encoded together. This should be a list of strings or a list of integers
                (tokenized string ids using the ``convert_tokens_to_ids`` method).
            text_pair (:obj:`str`, :obj:`List[str]` or :obj:`List[int]`, `optional`):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the ``tokenize`` method) or a list of integers (tokenized string ids using the
                ``convert_tokens_to_ids`` method).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        def get_input_ids(text: Union[List[int], str]):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    f"Input {text} is not valid. Should be a string, a list/tuple "
                    f"of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        input_ids = [get_input_ids(text) for text in texts]

        return self.prepare_for_model_many(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
            extended_token_type_ids=extended_token_type_ids,
        )

    def prepare_for_model_many(
        self,
        ids: List[List[int]],
        add_special_tokens: bool = True,
        padding_strategy: Union[bool, str, PaddingStrategy] = False,
        truncation_strategy: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        extended_token_type_ids: int = None,
    ) -> BatchEncoding:
        r"""
        Prepares a tuple of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            ids (:obj:`List[int]`):
                Tokenized input ids of the sequences. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
        """

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if truncation_strategy in (TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND):
            raise ValueError(
                f"truncation_strategy must be set to {TruncationStrategy.LONGEST_FIRST} or "
                f"{TruncationStrategy.DO_NOT_TRUNCATE} (got {truncation_strategy})"
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = self.get_encoding_length(ids, add_special_tokens=add_special_tokens)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, overflowing_tokens = self.truncate_many_sequences(
                ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        sequence = self.build_many_inputs_with_special_token(ids, add_special_tokens=add_special_tokens)
        token_type_ids = self.create_token_type_ids_from_many_sequences(
            ids, add_special_tokens=add_special_tokens, extended_token_type_ids=extended_token_type_ids
        )
    
        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        return BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

    def truncate_many_sequences(
        self,
        ids: List[List[int]],
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ) -> Tuple[List[List[int]], List[int]]:
        r"""
        Truncates a sequence pair in-place following the strategy.

        Args:
            ids (:obj:`List[List[int]]`):
                Tokenized input ids of the sequences. Can be obtained from a string by chaining the ``tokenize``
                and ``convert_tokens_to_ids`` methods.
            num_tokens_to_remove (:obj:`int`, `optional`, defaults to 0):
                Number of tokens to remove using the truncation strategy.
            truncation_strategy (:obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`,
                `optional`, defaults to :obj:`False`):
                The strategy to follow for truncation. Can be:

                * :obj:`'longest_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            stride (:obj:`int`, `optional`, defaults to 0):
                If set to a positive number, the overflowing tokens returned will contain some tokens from the main
                sequence returned. The value of this argument defines the number of additional tokens.

        Returns:
            :obj:`Tuple[List[List[int]], List[int]]`: The truncated ``ids`` and the list of overflowing tokens.
        """
        if num_tokens_to_remove <= 0:
            return ids, []

        if not isinstance(truncation_strategy, TruncationStrategy):
            truncation_strategy = TruncationStrategy(truncation_strategy)

        overflowing_tokens = []
        if truncation_strategy == TruncationStrategy.LONGEST_FIRST:
            for _ in range(num_tokens_to_remove):

                # Get the length of the longer sequence in ids
                lenghts = [len(i) for i in ids]
                longer_index = lenghts.index(max(lenghts))

                if not overflowing_tokens:
                    window_len = min(len(ids[longer_index]), stride + 1)
                else:
                    window_len = 1
                overflowing_tokens.extend(ids[longer_index][-window_len:])
                ids[longer_index] = ids[longer_index][:-1]

        return (ids, overflowing_tokens)

    @abstractmethod
    def build_many_inputs_with_special_token(
        self,
        token_ids: List[List[int]],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        r"""
        Build model inputs from a tuple of sequences for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - tuple of sequences: ``[CLS] A [SEP] B [SEP] ... [SEP]``

        Args:
            token_id (:obj:`List[List[int]]`):
                List of IDs to which the special tokens will be added.

        Returns:
            :obj:`List[List[int]]`: List of `input IDs <../glossary.html#input-ids>`__
            with the appropriate special tokens.
        """

    @abstractmethod
    def create_token_type_ids_from_many_sequences(
        self,
        token_ids: List[List[int]],
        add_special_tokens: bool = True,
        extended_token_type_ids: int = None,
    ) -> List[List[int]]:
        r"""
        Create a mask from the two sequences passed to be used in a sequences classification task. A BERT sequences
        mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence | third sequence | .... |

        Args:
            token_ids (:obj:`List[List[int]]`):
                List of list of IDs.

        Returns:
            :obj:`List[List[int]]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """

    @abstractmethod
    def get_encoding_length(self, token_ids: List[List[int]], add_special_tokens: bool = True) -> int:
        r"""
        Get the length of the encoding.

        Args:
            token_ids (:obj:`List[List[int]]`):
                List of list of IDs.

        Returns:
            :obj:`int`: Length of the encoding.
        """
