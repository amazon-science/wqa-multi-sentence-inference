from argparse import Namespace


class ExtendedNamespace(Namespace):
    r""" Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.

    This version is enhanced with dictionary capabilities.
    """

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    @classmethod
    def from_namespace(cls, other_namespace: Namespace):
        new = cls()
        new.__dict__ = other_namespace.__dict__
        return new

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        yield from self.__dict__.items()

    def __len__(self):
        return len(self.__dict__)
