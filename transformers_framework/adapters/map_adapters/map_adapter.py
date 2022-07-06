from abc import abstractmethod
from typing import Dict

from transformers_framework.adapters.transformer_adapter import TransformersAdapter


class MapAdapter(TransformersAdapter):
    r"""
    MapAdapters provide a map-like interface to retrieve data. Each subclass should
    override the __getitem__ and __len__ and __iter__ methods.
    """

    @abstractmethod
    def __getitem__(self, idx) -> Dict:
        r"""
        This function should use the arguments in `hyperparameters` to
        return a map over the (parsed) lines. This is the right place to return indexable data.

        >>> return self.data[idx]
        """

    @abstractmethod
    def __len__(self):
        r""" Returns the number of examples in the source dataset. """
