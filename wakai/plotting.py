"""Shared plotting configuration."""

from pathlib import Path

import matplotlib.pyplot as pl
from astropy.table import Table

from .config import PROJECT_ROOT


def load_plot_params(path=None):
    """Load the repository's Matplotlib defaults."""
    path = PROJECT_ROOT / "rcParams.txt" if path is None else Path(path)
    params = Table.read(path, format="csv")
    for i, name in enumerate(params['name']):
        try:
            pl.rcParams[name] = float(params['value'][i])
        except (TypeError, ValueError):
            pl.rcParams[name] = params['value'][i]
    return params
