"""Shared paths and package configuration."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
# Keep the historical string API while deriving it from the installed source tree.
DATA_PATH = str(PROJECT_ROOT / "data")
