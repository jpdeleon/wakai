"""Build GUI-ready stellar reference rows from SIMBAD and VizieR."""

from __future__ import annotations

import math
import re
from urllib.parse import quote

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .star import Star


DEFAULT_PARAMETER_SPECS = (
    {
        "id": "teff",
        "parameter": "Teff",
        "pattern": r"^(teff|t_eff|teff_val|teff_gspphot|teff50|teffspec)$",
    },
    {
        "id": "prot",
        "parameter": "Prot",
        "pattern": r"^(prot|p_rot|per_rot|rotper|rotation_period)$",
    },
    {
        "id": "liew",
        "parameter": "Li EW",
        "pattern": r"^(liew|li_?ew|ew_?li|ewli|wli)$",
    },
    {
        "id": "rhk",
        "parameter": "log R'HK",
        "pattern": r"^(log_?r.?hk|logrhk|r.?hk)$",
    },
    {
        "id": "feh",
        "parameter": "Fe/H",
        "pattern": r"^(feh|fe_h|\[fe/h\]|__fe_h_|m_h|metallicity)$",
    },
    # Astrometry and photometry below: inputs to the Cluster page (parallax/
    # pmra/pmdec/rv) and the Isochrones page (parallax + all six mags),
    # populated from Gaia DR3 + 2MASS at target-resolve time (see
    # server.py's /api/query/gaia) so every page's inputs are visible and
    # editable in one place instead of scattered across target metadata /
    # fetched fresh per page.
    {
        "id": "parallax",
        "parameter": "Parallax",
        "pattern": r"^(parallax|plx)$",
    },
    {
        "id": "pmra",
        "parameter": "pmRA",
        "pattern": r"^(pmra|pm_ra)$",
    },
    {
        "id": "pmdec",
        "parameter": "pmDec",
        "pattern": r"^(pmdec|pmde|pm_dec)$",
    },
    {
        "id": "rv",
        "parameter": "RV",
        "pattern": r"^(rv|radial_velocity|vrad)$",
    },
    {
        "id": "bpmag",
        "parameter": "BP mag",
        "pattern": r"^(bpmag|bp_mag|phot_bp_mean_mag)$",
    },
    {
        "id": "gmag",
        "parameter": "G mag",
        "pattern": r"^(gmag|g_mag|phot_g_mean_mag)$",
    },
    {
        "id": "rpmag",
        "parameter": "RP mag",
        "pattern": r"^(rpmag|rp_mag|phot_rp_mean_mag)$",
    },
    {
        "id": "jmag",
        "parameter": "J mag",
        "pattern": r"^(jmag|j_mag|j_m)$",
    },
    {
        "id": "hmag",
        "parameter": "H mag",
        "pattern": r"^(hmag|h_mag|h_m)$",
    },
    {
        "id": "kmag",
        "parameter": "K mag",
        "pattern": r"^(kmag|k_mag|ks_m|ks_mag)$",
    },
)


def json_scalar(value):
    """Convert catalog scalars to strict JSON values."""
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def empty_reference_row(spec):
    """Return a visible, non-fabricated placeholder for a required parameter."""
    return {
        "parameter": spec["parameter"],
        "value": None,
        "uncertainty": None,
        "reference": "No catalog measurement selected",
        "catalog": "",
        "id": spec["id"],
        "column": "",
        "unit": "",
        "source_url": "",
    }


def _candidate_row(record, parameter, parameter_id):
    row = {
        key: json_scalar(value)
        for key, value in record.items()
        if key not in {"key"}
    }
    row.update({
        "parameter": parameter,
        "id": parameter_id,
    })
    return row


def _candidate_sort_key(row):
    value_missing = row.get("value") is None
    separation = row.get("separation_arcsec")
    separation = float("inf") if separation is None else float(separation)
    return value_missing, separation, row.get("catalog", ""), row.get("column", "")


def query_vizier_reference(star: Star, parameter=None, radius=3):
    """Query VizieR and return default rows plus selectable source candidates."""
    tables = star.query_vizier(radius=radius, verbose=False)
    if not tables:
        return {
            "ref_table": [empty_reference_row(spec) for spec in DEFAULT_PARAMETER_SPECS],
            "candidates": [],
            "available_parameters": [],
            "tables": [],
        }

    table_summaries = []
    for key, table in zip(tables.keys(), tables.values()):
        metadata = getattr(table, "_meta", {})
        description = table.meta.get("description") or metadata.get("description") or str(key)
        table_summaries.append({
            "catalog": str(key),
            "reference": description,
            "source_url": (
                "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source="
                f"{quote(str(key), safe='/+')}"
            ),
            "rows": len(table),
        })

    available = star.query_vizier_param(param=None, radius=radius)

    if parameter:
        # The search box matches column names by substring with optional
        # `*` wildcards ("log" and "log*" both match logg/logTe/...). Escape
        # the input so regex metacharacters can't raise, then re-enable the
        # wildcard before handing it to query_vizier_param(use_regex=True).
        escaped = re.escape(parameter).replace(r"\*", ".*")
        records = star.query_vizier_param(
            param=escaped,
            radius=radius,
            use_regex=True,
            detailed=True,
        )
        parameter_id = re.sub(r"[^a-z0-9]+", "", parameter.lower())
        candidates = [
            _candidate_row(record, parameter, parameter_id)
            for record in records
        ]
        candidates.sort(key=_candidate_sort_key)
        return {
            "ref_table": [],
            "candidates": candidates,
            "available_parameters": available,
            "tables": table_summaries,
        }

    ref_table = []
    candidates = []
    for spec in DEFAULT_PARAMETER_SPECS:
        records = star.query_vizier_param(
            param=spec["pattern"],
            radius=radius,
            use_regex=True,
            detailed=True,
        )
        rows = [
            _candidate_row(record, spec["parameter"], spec["id"])
            for record in records
        ]
        rows.sort(key=_candidate_sort_key)
        candidates.extend(rows)
        ref_table.append(rows[0] if rows and rows[0].get("value") is not None else empty_reference_row(spec))

    return {
        "ref_table": ref_table,
        "candidates": candidates,
        "available_parameters": available,
        "tables": table_summaries,
    }


def _row_value(row, name):
    columns = {column.lower(): column for column in row.colnames}
    column = columns.get(name.lower())
    return json_scalar(row[column]) if column else None


def simbad_target_payload(star: Star):
    """Resolve a star with SIMBAD and return metadata suitable for the GUI."""
    result = star.query_simbad()
    if result is None or len(result) == 0:
        return None

    row = result[0]
    main_id = _row_value(row, "main_id") or star.star_name
    ra = _row_value(row, "ra")
    dec = _row_value(row, "dec")
    if isinstance(ra, str) or isinstance(dec, str):
        coord = SkyCoord(f"{ra} {dec}", unit=(u.hourangle, u.deg))
        ra, dec = coord.ra.deg, coord.dec.deg
    else:
        ra, dec = float(ra), float(dec)

    ids_value = _row_value(row, "ids") or ""
    identifiers = [identifier.strip() for identifier in str(ids_value).split("|") if identifier.strip()]
    gaia_id = None
    for release in ("DR3", "EDR3", "DR2"):
        match = re.search(rf"Gaia {release}\s+(\d+)", str(ids_value), re.IGNORECASE)
        if match:
            gaia_id = match.group(1)
            break

    return {
        "name": str(main_id).strip(),
        "query_name": star.star_name,
        "ra": float(ra),
        "dec": float(dec),
        "gaiaid": gaia_id,
        "object_type": _row_value(row, "otype"),
        "identifiers": identifiers,
        "source_url": (
            "https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
            f"{quote(star.star_name)}"
        ),
    }
