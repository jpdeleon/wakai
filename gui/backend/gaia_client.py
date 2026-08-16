"""Gaia TAP (ESA archive) query client, used by gui/backend/routes_query.py.

Split out of server.py so the Gaia HTTP/ADQL plumbing isn't tangled up with
Flask route handling -- these are plain functions with no Flask dependency
and no wakai package dependency, so this module needs no sys.path setup.
"""
import json
import urllib.error
import urllib.parse
import urllib.request

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

# 2.5/ln(10), the standard Gaia photometric error propagation constant:
# sigma_mag = MAG_ERR_CONST / (flux / flux_error).
MAG_ERR_CONST = 1.0857362047581294


class GaiaQueryError(Exception):
    """The Gaia TAP service was unreachable, timed out, or returned a
    malformed response -- distinct from a successful query that just found
    zero rows, so callers can report the two cases differently (502 vs 404)
    instead of collapsing both into a misleading "not found"."""


def _gaia_tap_query(adql: str) -> list[dict]:
    """Execute ADQL query against Gaia TAP service, return rows as dicts
    (possibly empty). Raises GaiaQueryError on network/timeout/malformed-
    response failures."""
    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "json",
        "QUERY": adql,
    })
    url = f"{GAIA_TAP_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        raise GaiaQueryError(f"Gaia TAP query failed: {e}") from e
    return data.get("data") or []


def _fetch_gaia_2mass_photometry(gaia_id):
    """Fetch parallax plus Gaia BP/G/RP and 2MASS J/H/Ks magnitudes
    (+errors) for the isochrones2 page's SingleStarModel fit, via Gaia's
    own official 2MASS crossmatch table (gaiadr3.tmass_psc_xsc_best_neighbour)
    -- one query, no separate VizieR round trip, and parallax comes from the
    same source row as the photometry rather than a second query. Returns
    {"photometry": {}, "parallax": None, "parallax_err": None} if the source
    isn't found; omits any individual band that's missing (no 2MASS match,
    or that band wasn't observed) rather than failing the whole fetch --
    SingleStarModel only needs whichever bands are actually available.
    """
    adql = f"""
        SELECT g.parallax, g.parallax_error,
               g.phot_bp_mean_mag, g.phot_bp_mean_flux_over_error,
               g.phot_g_mean_mag, g.phot_g_mean_flux_over_error,
               g.phot_rp_mean_mag, g.phot_rp_mean_flux_over_error,
               tm.j_m, tm.j_msigcom, tm.h_m, tm.h_msigcom, tm.ks_m, tm.ks_msigcom
        FROM gaiadr3.gaia_source AS g
        JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch ON g.source_id = xmatch.source_id
        JOIN gaiadr1.tmass_original_valid AS tm ON xmatch.original_ext_source_id = tm.designation
        WHERE g.source_id = {int(gaia_id)}
    """
    try:
        rows = _gaia_tap_query(adql)
    except GaiaQueryError:
        # This is a secondary enrichment step after the target's already
        # been resolved -- a transient Gaia TAP failure here shouldn't fail
        # the whole target-resolve request, same as "no 2MASS match found".
        rows = []
    if not rows:
        return {"photometry": {}, "parallax": None, "parallax_err": None}
    r = rows[0]

    def from_snr(mag, snr):
        if mag is None or snr is None or snr <= 0:
            return None
        return [float(mag), MAG_ERR_CONST / float(snr)]

    def from_err(mag, err):
        return [float(mag), float(err)] if mag is not None and err is not None else None

    photometry = {
        "BP": from_snr(r[2], r[3]),
        "G": from_snr(r[4], r[5]),
        "RP": from_snr(r[6], r[7]),
        "J": from_err(r[8], r[9]),
        "H": from_err(r[10], r[11]),
        "K": from_err(r[12], r[13]),
    }
    return {
        "photometry": {band: value for band, value in photometry.items() if value is not None},
        "parallax": float(r[0]) if r[0] is not None else None,
        "parallax_err": float(r[1]) if r[1] is not None else None,
    }
