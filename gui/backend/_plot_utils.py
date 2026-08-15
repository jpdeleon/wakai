"""Shared matplotlib-figure-to-data-URI helper for gui/backend/run_*.py
wrapper scripts that render plots (run_cluster.py, run_madys_cmd.py).

Kept separate from _stdio_guard.py, which every wrapper imports regardless
of whether it plots -- scripts running in matplotlib-free environments
(gyro-interp/BAFFLES/isochrones2 .venvs) must never pull in matplotlib.

Import this only after matplotlib.use("Agg") has already been called in the
importing script, same as the inline versions this replaces -- pyplot picks
a backend on its own first import, which must happen after use("Agg").
"""
import base64
import io

import matplotlib.pyplot as pl

PLOT_DPI = 140


def fig_to_data_uri(fig):
    """Serialize a Matplotlib figure to a base64 PNG data URI."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=PLOT_DPI, bbox_inches="tight")
    pl.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
