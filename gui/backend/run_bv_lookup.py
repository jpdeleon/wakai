#!/usr/bin/env python
"""Look up (B-V)o from Teff via the Pecaut & Mamajek dwarf table. Called via subprocess with JSON input on argv[1]."""
import json
import os
import sys

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

# Computed relative to this file rather than hardcoded so this works from any
# checkout of the repo (matches the pattern already used in run_cluster.py).
WAKAI_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WAKAI_PATH)

from wakai.catalog import get_bv_from_teff


def main():
    data = json.loads(sys.argv[1])

    result = get_bv_from_teff(
        teff=data["teff"],
        teff_err=data.get("teff_err"),
        feh=data.get("feh"),
    )
    result["status"] = "ok"
    print(json.dumps(result), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
