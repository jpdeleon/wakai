#!/usr/bin/env python
"""List every cluster name in a cluster catalog, for the Cluster page's
"cluster to plot" search box. Called via subprocess with JSON input on
argv[1]. Runs in the ``wakai`` environment (needs pandas/pyarrow).
"""
import json
import os
import sys

from _stdio_guard import redirect_stdout_to_stderr
REAL_STDOUT = redirect_stdout_to_stderr()

WAKAI_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WAKAI_PATH)

from wakai.cluster import load_cluster_catalog


def main():
    data = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
    catalog = data.get("catalog", "Perren2023")

    df = load_cluster_catalog(catalog=catalog)
    names = sorted(df["Name"].dropna().unique().tolist())

    print(json.dumps({"status": "ok", "catalog": catalog, "names": names}), file=REAL_STDOUT)


if __name__ == "__main__":
    main()
