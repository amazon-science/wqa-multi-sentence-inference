import hashlib
import json
import os
from collections.abc import Iterable
from functools import partial
from itertools import combinations
from string import ascii_uppercase
from typing import Any, Callable, Dict, Generator, List, Sequence, Union

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor
from transformers import PreTrainedTokenizer
from transformers.pipelines.base import pad_collate_fn
from transformers_lightning.utils import filter_generator


def bool_to_int(string):
    return int(string.lower().strip() in ('yes', 'pos', 'positive', '1', 'correct', 'true'))


def to_int(string):
    return int(string.lower().strip())


def to_float(string):
    return float(string.strip())


def _check_types(argument: str, types=[]):
    r""" Parse argument in one of the given types (in order) and return converted value. """
    for _type in types:
        try:
            if _type is bool:
                if argument.lower() not in ('true', 'false'):
                    raise ValueError()
                x = (argument.lower() == 'true')
            else:
                x = _type(argument)
            return x
        except ValueError:
            pass
    raise TypeError(f"Argument {argument} is not of allowed types: {types}")


def check_types(*types):
    r""" Parse argument in one of the given types (in order) and return converted value. """
    return partial(_check_types, types=types)


def split(_list: Sequence, part_length: int, drop_last: bool = False):
    r"""
    Split a list `_list` in parts of length `part_length`.
    Eventually drop last piece if it would have been shorter. """
    assert isinstance(part_length, int) and part_length > 0
    assert isinstance(_list, (list, tuple))

    res = []
    for i in range(0, len(_list), part_length):
        res.append(_list[i: i + part_length])
    
    if drop_last and len(res[-1]) < part_length:
        res = res[:-1]

    return res


def l2_norm(x, y, dim: int = -1, keepdim: bool = False, normalize: bool = True):  # noqa: E741
    r""" Computes L-Norm between two tensors on the given dimension. """
    if normalize:
        x = x / torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
        y = y / torch.linalg.norm(y, ord=2, dim=dim, keepdim=True)

    return (x - y).pow(2).sum(dim=dim, keepdim=keepdim).sqrt()


def _get_scattered_tensor(size: int, device: torch.device):
    indexes = list(zip(*[[x, x + 1] if x % 2 == 0 else [x, x - 1] for x in range(size)]))
    res = torch.zeros(size, size, dtype=torch.bool, device=device, requires_grad=False)
    res[indexes] = True
    return res


cache = {}

def get_scattered_tensor(size: int, device: torch.device, use_cache: bool = True):
    r""" Return a tensor (matrix) with the following True values:
    Example with size = 4
    0 1 0 0
    1 0 0 0
    0 0 0 1
    0 0 1 0
    """
    if use_cache is False:
        return _get_scattered_tensor(size, device)

    if size not in cache:
        cache[size] = _get_scattered_tensor(size, device)

    return cache[size]


def expand_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    logits = torch.stack([1 - probs, probs], dim=-1).log()
    return logits


def split_list(_list: List, length: int):
    r""" Split a list `l` in parts on length `length`. """
    assert length >= 1

    index = 0
    while True:
        yield _list[index:index + length]
        index += length
        if index >= len(_list):
            break


def get_rng_index(list_or_tuple) -> int:
    return torch.randint(0, len(list_or_tuple), size=()).item()


def write_dict_to_disk(_dict: Dict, folder_path, trainer: Trainer):
    r""" Write some dict to disk as key-value pairs. """
    os.makedirs(folder_path, exist_ok=True)
    for key, values in _dict.items():
        if isinstance(values, Tensor):
            values = extract_data_from_every_tensor(values)
        filename = os.path.join(folder_path, f"{key}-{trainer.global_rank}.tsv")
        with open(filename, "w") as fo:
            for line in values:
                if isinstance(line, (list, tuple)):
                    line = "\t".join(line)
                fo.write(f"{line}\n")


def shrink_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor = None,
    pad_token_id: int = 0,
):
    r""" Remove data on the sequence length dimension in the positions where every example is padded. """
    indexes = (input_ids != pad_token_id).any(dim=0)
    return (
        input_ids[..., indexes],
        attention_mask[..., indexes],
        token_type_ids[..., indexes] if token_type_ids is not None else None,
    )


def shrink_batch_dict(
    batch: Dict,
    pad_token_id: int = 0,
):
    r""" Remove data on the sequence length dimension in the positions where every example is padded. """
    indexes = (batch['input_ids'] != pad_token_id).any(dim=0)
    return {k: v[..., indexes] if v is not None else None for k, v in batch.items()}


def pad_sequence(sequence: List, padding_value: Any, length: int):
    r""" Pad a sequence with values up to length. """
    sequence += [padding_value] * (length - len(sequence))
    return sequence


def string_to_signature(string, length: int = 16):
    return hashlib.sha1(string.encode("utf-8")).hexdigest()[:length]


def special_zip(*iterators) -> Iterable:
    r""" Zip allowing None iterators (which will be threated as infinite None generators. """
    def inf_gen():
        while True:
            yield None
    iterators = (iter(iterator) if iterator is not None else inf_gen() for iterator in iterators)
    yield from zip(*iterators)


def none_if_all_none(iterable: Iterable) -> Union[Iterable, None]:
    r""" If all elements in iterable are None, return None, else return iterable. """
    if all(x is None for x in iterable):
        return None
    return iterable


def extract_data_from_every_tensor(tensor: Tensor):
    r""" Extract list of data from every kind of tensor on every device. """
    return tensor.cpu().detach().tolist()


def sample_from_distribution(logits: Tensor, sample_function: str = 'gumbel'):
    r"""
    Sample from generator logits either using gumbel distrib or multinomial distribution.
    Reimplement gumbel softmax because there is a bug in torch.nn.functional.gumbel_softmax
    when fp16 is used (https://github.com/pytorch/pytorch/issues/41663).
    Code taken from
    https://github.com/richarddwang/electra_pytorch/blob/9b2533e62cd1b6126feca323fb7b48480b8c2df0/pretrain.py#L318.
    Gumbel softmax is equal to what official ELECTRA code do,
    standard gumbel dist. = -ln(-ln(standard uniform dist.))
    """
    if sample_function == 'gumbel':
        loc = torch.tensor(0., device=logits.device, dtype=logits.dtype)
        scale = torch.tensor(1., device=logits.device, dtype=logits.dtype)
        gumbel_dist = torch.distributions.gumbel.Gumbel(loc, scale)
        return (logits + gumbel_dist.sample(logits.shape)).argmax(dim=-1)
    elif sample_function == 'multinomial':
        return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze()
    else:
        raise ValueError("`sample_function` not valid, choose between 'gumbel' and 'multinomial'")


def apply_to_generator(generator: Generator, function: Callable) -> Generator:
    r""" Apply a function to every element of a generator. """
    yield from (function(element) for element in generator)


def index_multi_tensors(*tensors: Sequence[Tensor], positions: Tensor = None):
    r""" Index many tensors where positions is True. """
    return (ten[positions] for ten in tensors)


def get_group_indexes_dict(indexes: Tensor) -> List[Tensor]:
    r"""
    Given an integer `torch.Tensor` `indexes`, return a `torch.Tensor` of indexes for each different value in
    `indexes`.

    Args:
        indexes: a `torch.Tensor`

    Return:
        A list of integer `torch.Tensor`s

    Example:
        >>> indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        >>> get_group_indexes(indexes)
        {0: tensor([0, 1, 2]), 1: tensor([3, 4, 5, 6])}
    """

    res: dict = {}
    for i, _id in enumerate(indexes):
        _id = _id.item()
        if _id in res:
            res[_id] += [i]
        else:
            res[_id] = [i]

    return {k: torch.tensor(x, dtype=torch.long) for k, x in res.items()}


def safe_value_to_list(integer, length):
    if isinstance(integer, Iterable):
        integer = list(integer)
        assert len(integer) == length
        return integer
    else:
        return [integer] * length


def _names_infinite_generator(prefix: str = '', postfix: str = ''):
    number_of_letters = 1
    while True:
        for el in combinations(ascii_uppercase, r=number_of_letters):
            el = prefix + ''.join(el) + postfix
            yield el
        number_of_letters += 1


def names_infinite_generator(prefix: str = '', postfix: str = '', process_id: int = None, world_size: int = None):
    generator = _names_infinite_generator(prefix=prefix, postfix=postfix)
    if process_id is not None and world_size is not None:
        return filter_generator(generator, step=world_size, offset=process_id)
    return generator


def collate_single_fn_with_exceptions(tokenizer: PreTrainedTokenizer, skip: List[str] = []) -> Callable:
    r"""
    Merge n dicts with identical keys creating list of value tensors.
    Do not convert original documents to tensors.
    """
    # convert values to tensors
    pad_collate_fn_instance = pad_collate_fn(tokenizer, None)

    def collate_fn(data: List[Dict]):
        process = pad_collate_fn_instance(data)
        return process

    return collate_fn


class Writer:

    def __init__(self, output_path: str, trainer: Trainer, chunk_size: int = 1000000):
        self.chunk_size = chunk_size
        self.output_path = output_path
        self.names_generator = names_infinite_generator(
            postfix='.jsonl', process_id=trainer.global_rank, world_size=trainer.world_size
        )
        self.start()

    def start(self):
        self.fo = open(os.path.join(self.output_path, next(self.names_generator)), "w")
        self.written = 0

    def reset(self):
        self.close()
        self.start()

    def close(self):
        self.fo.close()

    def write(self, data: Dict):
        self.fo.write(json.dumps(data) + "\n")
        self.written += 1

    def write_lines(self, lines_iterable: Iterable):
        for line in lines_iterable:
            if self.written == self.chunk_size:
                self.reset()
            self.write(line)
