"""Target-resolution routes: SIMBAD name resolution, VizieR reference-value
lookup, and Gaia DR3 crossmatch (the Target page's "Select a star" flow).
"""
import concurrent.futures
import os
import re
import sys
import threading
from collections import OrderedDict

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BACKEND_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from flask import Blueprint, jsonify, request  # noqa: E402

from wakai.reference import (  # noqa: E402
    DEFAULT_PARAMETER_SPECS,
    empty_reference_row,
    query_vizier_reference,
    simbad_target_payload,
)
from wakai.star import Star  # noqa: E402

import gaia_client  # noqa: E402
from gaia_client import GaiaQueryError, _fetch_gaia_2mass_photometry  # noqa: E402

query_bp = Blueprint("routes_query", __name__)

VIZIER_CACHE_MAX_TARGETS = 4
_VIZIER_TARGET_CACHE = OrderedDict()
# Guards _VIZIER_TARGET_CACHE's pop/insert/evict sequence, which isn't safe
# under concurrent requests now that the dev server runs threaded=True.
_VIZIER_CACHE_LOCK = threading.Lock()

# SIMBAD name resolution and Gaia DR3 crossmatch each round-trip to a remote
# archive that, independent of anything this process does, commonly takes
# several seconds per call (profiled: ~2.5s for SIMBAD, ~4s per Gaia TAP
# query) -- and query_gaia() below makes two of the latter. A target's
# resolved identity (coordinates, source_id, photometry, ...) doesn't change
# within a server's lifetime, so re-resolving the same target on every page
# load/navigation (applyRoute's triggerMockSearch path, see routing.js) pays
# that multi-second cost repeatedly for no new information. These caches
# make a second request for the same target near-instant, same LRU-by-
# insertion-order pattern as _VIZIER_TARGET_CACHE above.
_RESOLVE_CACHE_MAX = 64
_SIMBAD_CACHE = OrderedDict()
_SIMBAD_CACHE_LOCK = threading.Lock()
_GAIA_CACHE = OrderedDict()
_GAIA_CACHE_LOCK = threading.Lock()


def _cache_get(cache, lock, key):
    with lock:
        value = cache.pop(key, None)
        if value is not None:
            cache[key] = value  # move to most-recently-used position
        return value


def _cache_put(cache, lock, key, value, max_size):
    with lock:
        cache.pop(key, None)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)


def _default_parameters():
    return [empty_reference_row(spec) for spec in DEFAULT_PARAMETER_SPECS]


def _set_parameter_row(parameters, param_id, value, uncertainty, unit, reference, catalog, column, source_url):
    """Find `param_id`'s row (by DEFAULT_PARAMETER_SPECS id) in `parameters`
    and fill it in, if a real value was found -- looked up by id rather than
    a hardcoded list position so callers don't need to know row order."""
    row = next((r for r in parameters if r["id"] == param_id), None)
    if row is None or value is None:
        return row
    row.update({
        "value": value,
        "uncertainty": uncertainty,
        "reference": reference,
        "catalog": catalog,
        "column": column,
        "unit": unit,
        "source_url": source_url,
    })
    return row


def _cached_catalog_star(name, ra, dec, radius, gaia_id=None):
    """Reuse downloaded VizieR tables for follow-up parameter selections."""
    key = (round(float(ra), 7), round(float(dec), 7), round(float(radius), 3))
    with _VIZIER_CACHE_LOCK:
        star = _VIZIER_TARGET_CACHE.pop(key, None)
        if star is None:
            star = Star(
                name or str(gaia_id or "target"),
                ra_deg=float(ra),
                dec_deg=float(dec),
                gaiaid=gaia_id,
                verbose=False,
            )
        else:
            star.star_name = name or star.star_name
            star.gaiaid = gaia_id or star.gaiaid
        _VIZIER_TARGET_CACHE[key] = star
        while len(_VIZIER_TARGET_CACHE) > VIZIER_CACHE_MAX_TARGETS:
            _VIZIER_TARGET_CACHE.popitem(last=False)
    return star


@query_bp.route("/api/query/simbad", methods=["POST"])
def query_simbad():
    """Resolve a target name and return its SIMBAD identifiers and coordinates."""
    data = request.get_json(force=True)
    name = str(data.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "Provide a target name"}), 400

    cache_key = name.lower()
    cached = _cache_get(_SIMBAD_CACHE, _SIMBAD_CACHE_LOCK, cache_key)
    if cached is not None:
        return jsonify({"status": "ok", "target": cached})

    try:
        payload = simbad_target_payload(Star(name, verbose=False))
    except Exception as exc:
        return jsonify({"status": "error", "message": f"SIMBAD query failed: {exc}"}), 502
    if payload is None:
        return jsonify({"status": "error", "message": f"SIMBAD could not resolve {name}"}), 404
    _cache_put(_SIMBAD_CACHE, _SIMBAD_CACHE_LOCK, cache_key, payload, _RESOLVE_CACHE_MAX)
    return jsonify({"status": "ok", "target": payload})


@query_bp.route("/api/query/vizier", methods=["POST"])
def query_vizier():
    """Query VizieR near a target and build selectable reference rows."""
    data = request.get_json(force=True)
    name = str(data.get("name") or "").strip()
    ra = data.get("ra")
    dec = data.get("dec")
    parameter = str(data.get("parameter") or "").strip() or None
    try:
        radius = float(data.get("radius", 3))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Radius must be numeric"}), 400
    if radius <= 0 or radius > 60:
        return jsonify({"status": "error", "message": "Radius must be between 0 and 60 arcsec"}), 400

    if ra is None or dec is None:
        if not name:
            return jsonify({"status": "error", "message": "Provide target coordinates or a name"}), 400
        try:
            resolved = simbad_target_payload(Star(name, verbose=False))
        except Exception as exc:
            return jsonify({"status": "error", "message": f"SIMBAD query failed: {exc}"}), 502
        if resolved is None:
            return jsonify({"status": "error", "message": f"SIMBAD could not resolve {name}"}), 404
        ra, dec = resolved["ra"], resolved["dec"]

    try:
        star = _cached_catalog_star(name, ra, dec, radius, data.get("gaia_id"))
        result = query_vizier_reference(star, parameter=parameter, radius=radius)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"VizieR query failed: {exc}"}), 502

    return jsonify({
        "status": "ok",
        "target": {"name": star.star_name, "ra": star.ra_deg, "dec": star.dec_deg},
        **result,
    })


@query_bp.route("/api/query/gaia", methods=["POST"])
def query_gaia():
    """Query Gaia DR3 by source_id or target name."""
    data = request.get_json(force=True)
    name = str(data.get("name") or "").strip()
    gaia_id = str(data.get("gaia_id") or "").strip()
    ra = data.get("ra")
    dec = data.get("dec")

    # known_source_id is set whenever the numeric source_id is already known
    # from the request itself (rather than only discoverable from the main
    # query's result row) -- that's the common "reload/revisit a target
    # already resolved once" case (routing.js's numeric-gaia-id URL slug,
    # or triggerMockSearch after SIMBAD already returned a gaiaid), and lets
    # the two independent Gaia TAP round trips below (main astrometry query,
    # 2MASS photometry query) run concurrently instead of back-to-back,
    # since neither needs the other's result to start.
    known_source_id = None
    if gaia_id:
        if not gaia_id.isdigit():
            return jsonify({"status": "error", "message": "Gaia ID must be numeric"}), 400
        where = f"source_id = {gaia_id}"
        known_source_id = gaia_id
    elif ra is not None and dec is not None:
        try:
            ra, dec = float(ra), float(dec)
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "RA and Dec must be numeric"}), 400
        where = (
            "1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {3 / 3600}))"
        )
    elif name:
        # Try to resolve via Gaia DR3 or fallback to Simbad-like name search
        # Use a broad cone search or name match
        try:
            numeric_id = int(name)
            where = f"source_id = {numeric_id}"
            known_source_id = str(numeric_id)
        except ValueError:
            # Allowlist rather than blocklist a single quote: Gaia
            # designations only ever contain letters/digits/spaces/+-., so
            # stripping anything outside that set can't leave a stray quote
            # (or any other ADQL metacharacter) to close out of the LIKE
            # literal, unlike the previous approach of only removing "'".
            # Also strip ADQL LIKE wildcards (%, _) from user input so a
            # search term can't widen its own match pattern unexpectedly.
            safe_name = re.sub(r"[^A-Za-z0-9 +.\-]", "", name)
            where = f"UPPER(designation) LIKE UPPER('%{safe_name}%')"
    else:
        return jsonify({"status": "error", "message": "Provide 'name' or 'gaia_id'"}), 400

    cache_key = (
        f"id:{known_source_id}" if known_source_id
        else f"pos:{ra}:{dec}" if ra is not None and dec is not None
        else f"name:{name.lower()}"
    )
    cached = _cache_get(_GAIA_CACHE, _GAIA_CACHE_LOCK, cache_key)
    if cached is not None:
        return jsonify({"status": "ok", "target": cached})

    adql = f"""
        SELECT TOP 1 source_id, ra, dec, parallax, parallax_error,
               pmra, pmra_error, pmdec, pmdec_error,
               radial_velocity, radial_velocity_error,
               teff_gspphot, teff_gspphot_lower, teff_gspphot_upper
        FROM gaiadr3.gaia_source
        WHERE {where}
    """
    # Module-qualified (not `from gaia_client import _gaia_tap_query`) so
    # this call and _fetch_gaia_2mass_photometry's internal call share the
    # same patchable binding -- gaia_client._gaia_tap_query -- instead of
    # two independent name bindings that a single mock can't intercept both
    # of.
    if known_source_id is not None:
        # Run both TAP round trips concurrently -- see known_source_id's
        # comment above for why this is safe here (2MASS query only needs
        # the source_id, which is already known, not the main query's row).
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows_future = pool.submit(gaia_client._gaia_tap_query, adql)
            photometry_future = pool.submit(_fetch_gaia_2mass_photometry, known_source_id)
            try:
                rows = rows_future.result()
            except GaiaQueryError as e:
                return jsonify({"status": "error", "message": str(e)}), 502
            if not rows:
                return jsonify({"status": "error", "message": "No match found in Gaia DR3"}), 404
            photometry = photometry_future.result()["photometry"]
    else:
        try:
            rows = gaia_client._gaia_tap_query(adql)
        except GaiaQueryError as e:
            return jsonify({"status": "error", "message": str(e)}), 502
        if not rows:
            return jsonify({"status": "error", "message": "No match found in Gaia DR3"}), 404
        photometry = None  # fetched below once source_id is known

    r = rows[0]
    source_id = str(r[0])
    ra = r[1] if r[1] is not None else 0.0
    dec = r[2] if r[2] is not None else 0.0
    parallax = r[3] if r[3] is not None else None
    parallax_err = r[4] if r[4] is not None else None
    pmra = r[5] if r[5] is not None else None
    pmra_err = r[6] if r[6] is not None else None
    pmdec = r[7] if r[7] is not None else None
    pmdec_err = r[8] if r[8] is not None else None
    rv = r[9] if r[9] is not None else None
    rv_err = r[10] if r[10] is not None else None
    teff = r[11] if r[11] is not None else None
    teff_lo = r[12] if r[12] is not None else None
    teff_hi = r[13] if r[13] is not None else None

    teff_val = float(teff) if teff is not None else None
    teff_err = (
        (float(teff_hi) - float(teff_lo)) / 2
        if teff_hi is not None and teff_lo is not None
        else None
    )

    gaia_source_url = f"https://gea.esac.esa.int/archive/?source_id={source_id}"
    parameters = _default_parameters()
    _set_parameter_row(
        parameters, "teff", teff_val, teff_err, "K",
        "Gaia DR3 GSP-Phot", "Gaia DR3", "teff_gspphot", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "parallax", float(parallax) if parallax is not None else None,
        float(parallax_err) if parallax_err is not None else None, "mas",
        "Gaia DR3", "Gaia DR3", "parallax", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "pmra", float(pmra) if pmra is not None else None,
        float(pmra_err) if pmra_err is not None else None, "mas/yr",
        "Gaia DR3", "Gaia DR3", "pmra", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "pmdec", float(pmdec) if pmdec is not None else None,
        float(pmdec_err) if pmdec_err is not None else None, "mas/yr",
        "Gaia DR3", "Gaia DR3", "pmdec", gaia_source_url,
    )
    _set_parameter_row(
        parameters, "rv", float(rv) if rv is not None else None,
        float(rv_err) if rv_err is not None else None, "km/s",
        "Gaia DR3", "Gaia DR3", "radial_velocity", gaia_source_url,
    )

    # Photometry (Gaia BP/G/RP + 2MASS J/H/Ks via Gaia's own crossmatch
    # table) for the Isochrones page -- fetched here, once, at target-resolve
    # time, rather than re-fetched on every isochrones run. Already fetched
    # (in parallel with the query above) whenever known_source_id was set;
    # otherwise source_id was only just discovered from the row above, so
    # fetch it now.
    if photometry is None:
        photometry = _fetch_gaia_2mass_photometry(source_id)["photometry"]
    for band, param_id, unit, reference, catalog in (
        ("BP", "bpmag", "mag", "Gaia DR3", "Gaia DR3"),
        ("G", "gmag", "mag", "Gaia DR3", "Gaia DR3"),
        ("RP", "rpmag", "mag", "Gaia DR3", "Gaia DR3"),
        ("J", "jmag", "mag", "2MASS", "2MASS"),
        ("H", "hmag", "mag", "2MASS", "2MASS"),
        ("K", "kmag", "mag", "2MASS", "2MASS"),
    ):
        value = photometry.get(band)
        if value is not None:
            _set_parameter_row(
                parameters, param_id, value[0], value[1], unit,
                reference, catalog, band, gaia_source_url,
            )

    target = {
        "name": name or source_id,
        "gaiaid": source_id,
        "ra": float(ra),
        "dec": float(dec),
        "parallax": float(parallax) if parallax is not None else None,
        "parallax_error": float(parallax_err) if parallax_err is not None else None,
        "pmra": float(pmra) if pmra is not None else None,
        "pmdec": float(pmdec) if pmdec is not None else None,
        "rv": float(rv) if rv is not None else None,
        "parameters": parameters,
    }
    _cache_put(_GAIA_CACHE, _GAIA_CACHE_LOCK, cache_key, target, _RESOLVE_CACHE_MAX)
    # Also index by the resolved source_id -- a later request that already
    # knows the Gaia ID (e.g. routing.js's numeric-slug fast path) should
    # hit cache too, even if this particular request arrived by name/coords.
    if cache_key != f"id:{source_id}":
        _cache_put(_GAIA_CACHE, _GAIA_CACHE_LOCK, f"id:{source_id}", target, _RESOLVE_CACHE_MAX)
    return jsonify({"status": "ok", "target": target})
