"""Shared stdout guard for gui/backend/run_*.py wrapper scripts.

Each wrapper is invoked via subprocess.run() and must print exactly one JSON
object to stdout -- that's what server.py's _run_wrapper() parses. Several of
the external libraries these scripts import (numpy/pandas/astropy/emcee/
gyrointerp/BAFFLES/isochrones2) can print banners or warnings to stdout at
import or run time, which would land ahead of the JSON payload and break
that parse. Call redirect_stdout_to_stderr() before importing any such
library, and print the final JSON via the REAL_STDOUT it returns.
"""
import sys

REAL_STDOUT = sys.stdout


def redirect_stdout_to_stderr():
    """Send sys.stdout to stderr for the rest of the process; returns the
    real stdout so callers can do print(json.dumps(...), file=REAL_STDOUT)."""
    sys.stdout = sys.stderr
    return REAL_STDOUT
