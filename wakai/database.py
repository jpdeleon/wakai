"""SQLite persistence for targets, reference measurements, and model results."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def _sanitize_for_json(value):
    """Recursively replace NaN/Infinity with None.

    Posterior stats (e.g. an undefined gyro/BAFFLES/isochrone CI bound) can
    legitimately come back as NaN; json.dumps(..., allow_nan=False) below
    would otherwise raise ValueError mid-save and abort the whole target
    snapshot over one degenerate field.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS targets (
    id INTEGER PRIMARY KEY,
    target_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    gaia_id TEXT,
    ra REAL,
    dec REAL,
    parallax REAL,
    radial_velocity REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_measurements (
    id INTEGER PRIMARY KEY,
    target_id INTEGER NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    parameter_id TEXT,
    parameter TEXT NOT NULL,
    value REAL,
    uncertainty REAL,
    reference TEXT,
    catalog TEXT,
    column_name TEXT,
    unit TEXT,
    source_url TEXT
);

CREATE TABLE IF NOT EXISTS model_results (
    target_id INTEGER NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (target_id, method)
);

CREATE INDEX IF NOT EXISTS idx_reference_target
ON reference_measurements(target_id, position);
"""


def target_key(target):
    """Return a stable key, preferring Gaia ID over a normalized name."""
    gaia_id = str(target.get("gaiaid") or target.get("gaia_id") or "").strip()
    if gaia_id:
        return f"gaia:{gaia_id}"
    name = re.sub(r"[^a-z0-9]+", "-", str(target.get("name") or "").lower()).strip("-")
    if not name:
        raise ValueError("Target requires a Gaia ID or name")
    return f"name:{name}"


class TargetDatabase:
    """Small repository around a normalized SQLite schema."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(SCHEMA)
            connection.commit()

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save(self, target, ref_table, results):
        key = target_key(target)
        now = datetime.now(timezone.utc).isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO targets (
                    target_key, name, gaia_id, ra, dec, parallax,
                    radial_velocity, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_key) DO UPDATE SET
                    name = excluded.name,
                    gaia_id = excluded.gaia_id,
                    ra = excluded.ra,
                    dec = excluded.dec,
                    parallax = excluded.parallax,
                    radial_velocity = excluded.radial_velocity,
                    updated_at = excluded.updated_at
                """,
                (
                    key,
                    str(target.get("name") or target.get("gaiaid") or "target"),
                    str(target.get("gaiaid") or target.get("gaia_id") or "") or None,
                    target.get("ra"),
                    target.get("dec"),
                    target.get("parallax"),
                    target.get("rv", target.get("radial_velocity")),
                    now,
                    now,
                ),
            )
            target_id = connection.execute(
                "SELECT id FROM targets WHERE target_key = ?", (key,)
            ).fetchone()["id"]

            connection.execute(
                "DELETE FROM reference_measurements WHERE target_id = ?", (target_id,)
            )
            connection.executemany(
                """
                INSERT INTO reference_measurements (
                    target_id, position, parameter_id, parameter, value,
                    uncertainty, reference, catalog, column_name, unit, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        target_id,
                        position,
                        row.get("id"),
                        str(row.get("parameter") or ""),
                        row.get("value"),
                        row.get("uncertainty"),
                        row.get("reference"),
                        row.get("catalog"),
                        row.get("column"),
                        row.get("unit"),
                        row.get("source_url"),
                    )
                    for position, row in enumerate(ref_table)
                ],
            )
            connection.execute("DELETE FROM model_results WHERE target_id = ?", (target_id,))
            connection.executemany(
                "INSERT INTO model_results (target_id, method, payload_json) VALUES (?, ?, ?)",
                [
                    (target_id, method, json.dumps(_sanitize_for_json(payload), allow_nan=False))
                    for method, payload in results.items()
                ],
            )
            connection.commit()
        return key

    def load(self, *, gaia_id=None, name=None, key=None):
        keys = [key] if key else []
        if not keys and gaia_id:
            keys.append(target_key({"gaiaid": gaia_id}))
        if name:
            name_key = target_key({"name": name})
            if name_key not in keys:
                keys.append(name_key)
        if not keys:
            raise ValueError("Provide gaia_id, name, or key")
        with closing(self.connect()) as connection:
            target_row = None
            for lookup_key in keys:
                target_row = connection.execute(
                    "SELECT * FROM targets WHERE target_key = ?", (lookup_key,)
                ).fetchone()
                if target_row is not None:
                    break
            if target_row is None:
                return None

            measurements = connection.execute(
                """
                SELECT parameter_id, parameter, value, uncertainty, reference,
                       catalog, column_name, unit, source_url
                FROM reference_measurements
                WHERE target_id = ?
                ORDER BY position
                """,
                (target_row["id"],),
            ).fetchall()
            result_rows = connection.execute(
                "SELECT method, payload_json FROM model_results WHERE target_id = ?",
                (target_row["id"],),
            ).fetchall()

        return {
            "key": target_row["target_key"],
            "target": {
                "name": target_row["name"],
                "gaiaid": target_row["gaia_id"],
                "ra": target_row["ra"],
                "dec": target_row["dec"],
                "parallax": target_row["parallax"],
                "rv": target_row["radial_velocity"],
            },
            "ref_table": [
                {
                    "id": row["parameter_id"],
                    "parameter": row["parameter"],
                    "value": row["value"],
                    "uncertainty": row["uncertainty"],
                    "reference": row["reference"] or "",
                    "catalog": row["catalog"] or "",
                    "column": row["column_name"] or "",
                    "unit": row["unit"] or "",
                    "source_url": row["source_url"] or "",
                }
                for row in measurements
            ],
            "results": {
                row["method"]: json.loads(row["payload_json"])
                for row in result_rows
            },
            "updated_at": target_row["updated_at"],
        }

    def list_targets(self):
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT target_key, name, gaia_id, updated_at
                FROM targets ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]
