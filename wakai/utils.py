"""Small dependency-free helpers."""

import itertools


def flatten_list(lol):
    """flatten list of list (lol)"""
    return list(itertools.chain.from_iterable(lol))
