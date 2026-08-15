"""Cross-match the TOI catalog against external catalogs by TIC ID.

CLI entry point (registered in ``pyproject.toml``)::

    uv run wakai.crossmatch_toi --catalog gaiaDR3
    uv run wakai.crossmatch_toi --catalog gaiaDR3 --match_method position
    uv run wakai.crossmatch_toi --catalog Perren2023
    uv run wakai.crossmatch_toi --catalog Hunt2023
    uv run wakai.crossmatch_toi --catalog Hunt2024
    uv run wakai.crossmatch_toi --catalog CantatGaudin2020
    uv run wakai.crossmatch_toi --catalog Ratzenboeck2023

``gaiaDR3`` cross-matches TOIs against Gaia DR3 by one of two strategies,
selected with ``--match_method`` (also used by every other ``--catalog`` and
by ``--merge`` below, to pick which of the two files they join onto):

- ``id`` (default): TOI -> TIC -> Gaia DR3, saved to
  ``data/TOIs_GaiaDR3_xmatch_id.csv``. For every TIC ID in the TOI catalog,
  the TIC catalog's linked Gaia DR3 ``source_id`` is looked up (queried in
  chunks, see ``_TIC_QUERY_CHUNK_SIZE``), then radial velocity and GSP-Phot
  Teff are pulled from the Gaia archive for those source IDs. Inherits
  whatever cross-ID the TIC team computed at TIC construction time, with no
  per-row match-quality signal, and a small fraction of TIC rows genuinely
  have no Gaia link at all.
- ``position``: TOI sky position -> Gaia DR3, saved to
  ``data/TOIs_GaiaDR3_xmatch_position.csv``, via ``astroquery.xmatch``.
  Independent of the TIC catalog's own cross-ID, and keeps the nearest
  candidate (smallest ``angDist``) per TOI as an inspectable match-quality
  signal. TOI/TIC coordinates are reported at epoch J2000.0 (Stassun et al.
  2019, TIC-8) while Gaia DR3 astrometry is at epoch J2016.0, so positions
  are forward-propagated with each target's TIC proper motion before the
  search (see ``_propagate_to_gaia_epoch``) — skipping this would bias
  high-proper-motion targets, which is the scenario this cross-match cares
  most about avoiding in crowded cluster fields. Agrees with ``id`` on the
  same Gaia source for ~99.9% of TOIs matched by both.

Gaia DR3 ``source_id`` values are ~19-digit integers, well beyond float64's
2**53 exact-integer range, and ``astropy.table.Table.to_pandas()`` silently
loses precision for *masked* int64 columns (it round-trips through float64
internally). The Gaia ID column is therefore built directly from the raw
astropy column instead of trusting a blanket ``.astype(float)``/pandas
conversion, which is what the original exploratory notebook
(``notebooks/crossmatch_TOI_with_GaiaDR3.ipynb``) did.

``Perren2023``, ``Hunt2023``, ``Hunt2024``, ``CantatGaudin2020``, and
``Ratzenboeck2023`` all join a cluster catalog's member table onto whichever
``gaiaDR3`` cross-match ``--match_method`` selects, by exact Gaia
``source_id`` — i.e. they only report TOIs whose host star is already a
listed member of that cluster catalog. Each writes its own
``--match_method``-suffixed output file (e.g.
``Hunt2023/TOIs_in_Hunt2023_id.csv`` vs. ``..._position.csv``) so switching
``--match_method`` doesn't return a stale cache from the other method.

``Perren2023`` additionally requires ``data/Perren2023/UCC_members.parquet``
to already be downloaded (see
``notebooks/crossmatch_TOI_with_Perren2023.ipynb`` for the Zenodo download
cells); this command does not fetch it automatically since the record id is
a manually-tracked catalog version.

``Ratzenboeck2023`` ships as two linked Vizier tables sharing a ``SigMA``
group id: 2023a has per-star membership (Gaia id, no group name) and 2023b
has per-group names and ages. The exploratory notebook downloaded both but
never joined them, so its saved output has no cluster name or age at all;
this command joins member -> group name -> group age on ``SigMA`` to fix
that.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import typer
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astroquery.xmatch import XMatch
from loguru import logger

from .cluster import load_cluster_members
from .config import DATA_PATH
from .planet import get_tois
from .vizier import CatalogDownloader

app = typer.Typer(add_completion=False, help="Cross-match TOIs against external catalogs.")

TIC_COLUMNS = ["ID", "ra", "dec", "plx", "pmRA", "pmDEC", "Teff"]
GAIA_RV_COLUMNS = ["radial_velocity", "radial_velocity_error", "teff_gspphot"]

#: Max TIC IDs per ``Catalogs.query_criteria`` call. Sending the whole TOI
#: TIC-ID list (~7800 ids) as a single ``ID=[...]`` filter makes MAST's
#: Discovery Portal API silently drop a fraction of the matches scattered
#: throughout the request (confirmed empirically: one query for ~7820 ids
#: returned only ~79% of them, but re-querying exactly the missing ids as
#: their own batch of ~1700 returned 100%) -- no error, no partial-page
#: signal, just missing rows. Chunking keeps each request's ID list well
#: under whatever server-side limit triggers this.
_TIC_QUERY_CHUNK_SIZE = 1000

#: TOI/TIC ``ra_deg``/``dec_deg`` reference epoch (Stassun et al. 2019,
#: TIC-8: TIC astrometry is reported at epoch=2000.0) and Gaia DR3's own
#: astrometric reference epoch (Gaia Collaboration et al. 2022), used to
#: forward-propagate TOI positions before the position-based cross-match.
TOI_POSITION_EPOCH = Time(2000.0, format="jyear")
GAIA_DR3_EPOCH = Time(2016.0, format="jyear")

#: Position cross-match search radius (astroquery.xmatch max_distance).
POSITION_XMATCH_RADIUS = 5 * u.arcsec

#: Per-catalog raw member-table column -> canonical name. pandas silently
#: no-ops a rename key that isn't present in the frame, so these are kept
#: catalog-specific (matching each Vizier table's real layout) rather than
#: one shared dict where some entries would be dead for a given catalog.
_CLUSTER_MEMBER_RENAME: dict[str, dict[str, str]] = {
    "Hunt2023": {
        "name": "Name",
        "Source": "GaiaDR3",
        "probs": "Prob",
        "RA_ICRS": "ra",
        "DE_ICRS": "dec",
        "pmRA": "pmra",
        "pmDE": "pmdec",
        "Plx": "parallax",
        "RV": "radial_velocity",
    },
    "Hunt2024": {
        # Hunt2024's member table already names these Name/GaiaDR3/Prob.
        "RA_ICRS": "ra",
        "DE_ICRS": "dec",
        "pmRA": "pmra",
        "pmDE": "pmdec",
        "Plx": "parallax",
        "RV": "radial_velocity",
    },
    "CantatGaudin2020": {
        "Cluster": "Name",
        "Source": "GaiaDR3",
        "Proba": "Prob",
        "RA_ICRS": "ra",
        "DE_ICRS": "dec",
        "pmRA": "pmra",
        "pmDE": "pmdec",
        "Plx": "parallax",
        "RV": "radial_velocity",
    },
}

#: Output columns to keep, if present, for the Hunt2023/Hunt2024/CantatGaudin2020/
#: Ratzenboeck2023 cluster cross-matches. A superset of what the exploratory
#: notebooks selected per catalog; columns absent for a given catalog are
#: silently dropped (see crossmatch_cluster_catalog), so one shared list is
#: used instead of hand-tuning it per catalog.
_CLUSTER_OUTPUT_COLUMNS = (
    "TOI\tTIC ID\tName\tGaiaDR3\tProb\tTESS Disposition\tTFOPWG Disposition\tComments\tTESS Mag\t"
    "TESS Mag err\tSource_1\tDetection\tEpoch (BJD)\tEpoch (BJD) err\tPeriod (days)\tPeriod (days) err\t"
    "Duration (hours)\tDuration (hours) err\tDepth (mmag)\tDepth (mmag) err\tDepth (ppm)\tDepth (ppm) err\t"
    "Planet Radius (R_Earth)\tPlanet Radius (R_Earth) err\tPlanet Insolation (Earth Flux)\t"
    "Planet Equil Temp (K)\tPlanet SNR\tStellar Distance (pc)\tStellar Distance (pc) err\t"
    "Stellar Eff Temp (K)\tStellar Eff Temp (K) err\tStellar log(g) (cm/s^2)\tStellar log(g) (cm/s^2) err\t"
    "Stellar Radius (R_Sun)\tStellar Radius (R_Sun) err\tStellar Metallicity\tStellar Metallicity err\t"
    "Stellar Mass (M_Sun)\tStellar Mass (M_Sun) err\tSectors\tDate TOI Alerted (UTC)\t"
    "Date TOI Updated (UTC)\tDate Modified\tra_deg\tdec_deg\tPipeline Signal ID\tRA\tDec\t"
    "PM RA (mas/yr)\tPM RA err (mas/yr)\tPM Dec (mas/yr)\tPM Dec err (mas/yr)\tDR3Name\tRAdeg\tDEdeg\t"
    "errHalfMaj\terrHalfMin\terrPosAng\tSolID\tSource_2\tRandomI\te_RAdeg\te_DEdeg\te_Plx\tRPlx\tPM\t"
    "e_pmRA\te_pmDE\tRADEcor\tRAPlxcor\tRApmRAcor\tRApmDEcor\tDEPlxcor\tDEpmRAcor\tDEpmDEcor\t"
    "PlxpmRAcor\tPlxpmDEcor\tpmRApmDEcor\tNAL\tNAC\tNgAL\tNbAL\tgofAL\tchi2AL\tepsi\tsepsi\tSolved\t"
    "APF\tnueff\tMatchObsA\tNper\tamax\tMatchObs\tIPDgofha\tIPDgofhp\tIPDfmp\tIPDfow\tRUWE\tDup\t"
    "o_Gmag\tFG\te_FG\tRFG\tGmag_y\te_Gmag\to_BPmag\tFBP\te_FBP\tRFBP\te_BPmag\to_RPmag\tFRP\te_FRP\t"
    "RFRP\te_RPmag\tE(BP/RP)\tNBPcont\tNBPblend\tNRPcont\tNRPblend\tMode\tBP-G\tG-RP\te_RV\tn_RV\t"
    "o_RV\to_RVd\tRVNper\tRVS/N\tRVgof\tRVchi2\tRVTdur\tRVamp\tRVtempTeff\tRVtemplogg\tRVtemp[Fe/H]\t"
    "Vatmparam\tvbroad\te_Vbroad\to_Vbroad\tGRVSmag\te_GRVSmag\to_GRVSmag\tRVSS/N\tVarFlag\tPQSO\t"
    "PGal\tPSS\tTeff\tb_Teff\tB_Teff\tlogg\tb_logg\tB_logg\t[Fe/H]\tb_[Fe/H]\tB_[Fe/H]\tDist\tb_Dist\t"
    "B_Dist\tA0\tb_A0\tB_A0\tAG\tb_AG\tB_AG\tE(BP-RP)\tb_E(BP-RP)\tB_E(BP-RP)\tLib\tRAJ2000\tDEJ2000\t"
    "e_RAJ2000\te_DEJ2000\tRADEcorJ2000\tPrevious CTOI\tMaster\tSG1A\tSG1B\tSG2\tSG3\tSG4\tSG5\tESM\t"
    "TSM\tPredicted Mass (M_Earth)\tTime Series Observations\tSpectroscopy Observations\t"
    "Imaging Observations"
).split("\t")


class Catalog(str, Enum):
    """External catalogs supported by this command."""

    gaiaDR3 = "gaiaDR3"
    Perren2023 = "Perren2023"
    Hunt2023 = "Hunt2023"
    Hunt2024 = "Hunt2024"
    CantatGaudin2020 = "CantatGaudin2020"
    Ratzenboeck2023 = "Ratzenboeck2023"


class MatchMethod(str, Enum):
    """TOI -> Gaia DR3 matching strategy for ``--catalog gaiaDR3``."""

    id = "id"
    position = "position"


def _propagate_to_gaia_epoch(tois: pd.DataFrame) -> pd.DataFrame:
    """Forward-propagate TOI ``ra_deg``/``dec_deg`` to Gaia DR3's epoch.

    Skipping this before a position cross-match biases high-proper-motion
    targets: over the ~16-year baseline between the TOI/TIC epoch (J2000.0)
    and Gaia DR3's epoch (J2016.0), a target moving at 500 mas/yr drifts by
    >8", well outside a 5" search radius — and in the crowded cluster fields
    this package targets, a smaller drift can still be enough to lock onto
    the wrong neighboring star instead of the true counterpart. Proper motion
    only is applied (no parallax/radial_velocity term): the TOI table has no
    radial velocity, and at arcsecond search-radius precision the
    perspective-acceleration correction a distance term would add is
    negligible. Targets with no TIC proper motion are left at their catalog
    position.

    Parameters
    ----------
    tois : pd.DataFrame
        TOI catalog with ``ra_deg``, ``dec_deg``, ``PM RA (mas/yr)``, and
        ``PM Dec (mas/yr)`` columns (as returned by ``get_tois``).

    Returns
    -------
    pd.DataFrame
        Copy of ``tois`` with ``ra_deg_gaia_epoch``/``dec_deg_gaia_epoch``
        columns added.
    """
    pmra = tois["PM RA (mas/yr)"].to_numpy(dtype=float)
    pmdec = tois["PM Dec (mas/yr)"].to_numpy(dtype=float)
    has_pm = ~np.isnan(pmra) & ~np.isnan(pmdec)

    coords = SkyCoord(
        ra=tois["ra_deg"].to_numpy(dtype=float) * u.deg,
        dec=tois["dec_deg"].to_numpy(dtype=float) * u.deg,
        pm_ra_cosdec=np.where(has_pm, pmra, 0.0) * u.mas / u.yr,
        pm_dec=np.where(has_pm, pmdec, 0.0) * u.mas / u.yr,
        obstime=TOI_POSITION_EPOCH,
    )
    propagated = coords.apply_space_motion(new_obstime=GAIA_DR3_EPOCH)

    out = tois.copy()
    out["ra_deg_gaia_epoch"] = np.where(has_pm, propagated.ra.deg, out["ra_deg"])
    out["dec_deg_gaia_epoch"] = np.where(has_pm, propagated.dec.deg, out["dec_deg"])
    return out


def _query_tic_with_gaia_ids(tic_ids: np.ndarray, verbose: bool = True) -> pd.DataFrame:
    """Query the TIC catalog for astrometry plus the linked Gaia DR3 source_id.

    Queried in chunks of `_TIC_QUERY_CHUNK_SIZE` rather than all at once.
    A single ``Catalogs.query_criteria(catalog="TIC", ID=[...])`` call with
    the full TOI TIC-ID list (~7800 ids) was found to silently drop ~21% of
    the requested ids -- no error, no partial-result flag, just missing rows
    -- dropping the ``id`` match method's apparent match rate to ~79% of
    TOIs. The dropped ids were scattered throughout the request rather than
    clustered at the end, ruling out a simple truncation-at-N-results
    explanation; re-querying exactly the missing ~1700 ids as their own
    single batch returned 100% of them, confirming MAST's Discovery Portal
    API is unreliable for an ``ID IN (...)`` filter this large rather than
    those TIC rows genuinely lacking data. Chunking to `_TIC_QUERY_CHUNK_SIZE`
    ids per call resolved all ~7800/7800 ids in testing.

    Parameters
    ----------
    tic_ids : np.ndarray
        Unique TIC IDs to query.
    verbose : bool
        Log per-chunk progress.

    Returns
    -------
    pd.DataFrame
        One row per TIC ID with columns ``TIC_COLUMNS`` plus ``GAIA``
        (nullable ``Int64``, ``pd.NA`` where the TIC catalog has no Gaia
        cross-ID).
    """
    tic_ids = list(tic_ids)
    chunks = [tic_ids[i : i + _TIC_QUERY_CHUNK_SIZE] for i in range(0, len(tic_ids), _TIC_QUERY_CHUNK_SIZE)]

    frames = []
    for i, chunk in enumerate(chunks, start=1):
        if verbose:
            logger.info(f"Querying TIC catalog chunk {i}/{len(chunks)} ({len(chunk)} ids)")
        tic_table = Catalogs.query_criteria(catalog="TIC", ID=chunk)
        chunk_df = tic_table.to_pandas()[TIC_COLUMNS].astype(float)
        chunk_df["GAIA"] = pd.array(
            [int(gid) if gid is not np.ma.masked else pd.NA for gid in tic_table["GAIA"]],
            dtype="Int64",
        )
        frames.append(chunk_df)

    tic_df = pd.concat(frames, ignore_index=True)
    return tic_df[["ID", "GAIA", "ra", "dec", "plx", "pmRA", "pmDEC", "Teff"]]


def _query_gaia_rv_teff(gaia_ids: list[int]) -> pd.DataFrame:
    """Query Gaia DR3 for radial velocity and GSP-Phot Teff for source_ids."""
    ids_str = ",".join(str(int(gid)) for gid in gaia_ids)
    query = f"""
    SELECT source_id, radial_velocity, radial_velocity_error, teff_gspphot
    FROM gaiadr3.gaia_source
    WHERE source_id IN ({ids_str})
    """
    job = Gaia.launch_job_async(query)
    rv_df = job.get_results().to_pandas().rename(columns={"source_id": "GAIA", "SOURCE_ID": "GAIA"})
    rv_df["GAIA"] = rv_df["GAIA"].astype("Int64")
    for col in GAIA_RV_COLUMNS:
        rv_df[col] = rv_df[col].astype(float)
    return rv_df


def crossmatch_gaiaDR3(
    match_method: MatchMethod = MatchMethod.id,
    clobber: bool = False,
    outdir: str = DATA_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """TOI -> Gaia DR3 cross-match, by TIC cross-ID or by sky position.

    Parameters
    ----------
    match_method : MatchMethod
        ``id`` (default) resolves Gaia DR3 ``source_id`` via the TIC
        catalog's own cross-ID (see `_crossmatch_gaiaDR3_by_id`); ``position``
        cross-matches TOI sky positions directly against Gaia DR3, with
        proper-motion epoch propagation (see `_crossmatch_gaiaDR3_by_position`).
    clobber : bool
        Re-download/re-query and overwrite an existing output file.
    outdir : str
        Directory to save the cross-match CSV to.
    verbose : bool
        Log progress.

    Returns
    -------
    pd.DataFrame
        TOI rows merged with their Gaia DR3 match.
    """
    if match_method is MatchMethod.position:
        return _crossmatch_gaiaDR3_by_position(clobber=clobber, outdir=outdir, verbose=verbose)
    return _crossmatch_gaiaDR3_by_id(clobber=clobber, outdir=outdir, verbose=verbose)


def _crossmatch_gaiaDR3_by_id(
    clobber: bool = False,
    outdir: str = DATA_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """ID-based TOI -> Gaia DR3 cross-match via the TIC catalog's Gaia link.

    Parameters
    ----------
    clobber : bool
        Re-download the TOI catalog and overwrite an existing output file.
    outdir : str
        Directory to save ``TOIs_GaiaDR3_xmatch_id.csv`` to.
    verbose : bool
        Log progress.

    Returns
    -------
    pd.DataFrame
        TOI rows merged with TIC astrometry and Gaia DR3 radial velocity/Teff.
    """
    fp = Path(outdir, "TOIs_GaiaDR3_xmatch_id.csv")
    if fp.exists() and not clobber:
        if verbose:
            logger.info(f"Loaded: {fp} (use --clobber to re-query and overwrite)")
        return pd.read_csv(fp, dtype={"GaiaDR3": "Int64"})

    tois = get_tois(clobber=clobber, verbose=verbose)

    if verbose:
        logger.info(f"Querying TIC catalog for {tois['TIC ID'].nunique()} unique TIC IDs")
    tic_df = _query_tic_with_gaia_ids(tois["TIC ID"].unique(), verbose=verbose)

    gaia_ids = tic_df["GAIA"].dropna().unique().tolist()
    if verbose:
        logger.info(f"Querying Gaia DR3 for {len(gaia_ids)} linked source_ids")
    rv_df = _query_gaia_rv_teff(gaia_ids)

    tic_df = pd.merge(tic_df, rv_df, on="GAIA", how="left")
    tic_df["TIC ID"] = tic_df["ID"].astype(int)

    matches = pd.merge(tois, tic_df, on="TIC ID", how="inner")
    matches = matches.rename(
        columns={
            "GAIA": "GaiaDR3",
            "plx": "parallax",
            "pmRA": "pmra",
            "pmDEC": "pmdec",
            "Teff": "Teff_TIC",
            "teff_gspphot": "Teff_gaia",
        }
    )

    Path(outdir).mkdir(parents=True, exist_ok=True)
    matches.to_csv(fp, index=False)
    if verbose:
        logger.info(f"Saved: {fp}")

    return matches


def _crossmatch_gaiaDR3_by_position(
    clobber: bool = False,
    outdir: str = DATA_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Position-based TOI -> Gaia DR3 cross-match via ``astroquery.xmatch``.

    Cross-matches each TOI's sky position -- epoch-propagated to Gaia DR3's
    J2016.0 reference epoch by `_propagate_to_gaia_epoch` -- against Gaia DR3
    within `POSITION_XMATCH_RADIUS`, keeping the nearest candidate (smallest
    ``angDist``) per TOI. Independent of `_crossmatch_gaiaDR3_by_id`: it does
    not rely on the TIC catalog's own Gaia cross-ID, so it can recover a Gaia
    source for TOIs where that link is missing, and disagreements between the
    two methods are a useful signal that the TIC cross-ID is wrong.

    Parameters
    ----------
    clobber : bool
        Re-download the TOI catalog and overwrite an existing output file.
    outdir : str
        Directory to save ``TOIs_GaiaDR3_xmatch_position.csv`` to.
    verbose : bool
        Log progress.

    Returns
    -------
    pd.DataFrame
        One row per TOI with its nearest Gaia DR3 position match.
    """
    fp = Path(outdir, "TOIs_GaiaDR3_xmatch_position.csv")
    if fp.exists() and not clobber:
        if verbose:
            logger.info(f"Loaded: {fp} (use --clobber to re-query and overwrite)")
        return pd.read_csv(fp, dtype={"GaiaDR3": "Int64"})

    tois = get_tois(clobber=clobber, verbose=verbose)
    tois = _propagate_to_gaia_epoch(tois)

    if verbose:
        logger.info(
            f"Position cross-matching {len(tois)} TOIs against Gaia DR3 "
            f"(epoch-propagated to J2016.0, {POSITION_XMATCH_RADIUS} radius)"
        )
    gaia_table = XMatch.query(
        cat1=Table.from_pandas(tois),
        cat2="vizier:I/355/gaiadr3",
        max_distance=POSITION_XMATCH_RADIUS,
        colRA1="ra_deg_gaia_epoch",
        colDec1="dec_deg_gaia_epoch",
    )

    matches = gaia_table.to_pandas()
    # Keep the nearest Gaia DR3 candidate (smallest angDist) per TOI instead
    # of every candidate within the search radius.
    matches = matches.sort_values("angDist").drop_duplicates(subset="TOI", keep="first").sort_index()
    matches["GaiaDR3"] = matches["DR3Name"].apply(lambda x: int(x.split()[-1])).astype("Int64")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    matches.to_csv(fp, index=False)
    if verbose:
        logger.info(f"Saved: {fp}")

    return matches


def crossmatch_perren2023(
    match_method: MatchMethod = MatchMethod.id,
    clobber: bool = False,
    outdir: str = DATA_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """TOI -> Perren+2023 (UCC) cluster cross-match by Gaia DR3 source_id.

    Parameters
    ----------
    match_method : MatchMethod
        Which ``gaiaDR3`` cross-match (``id`` or ``position``) to join onto
        the UCC member list; see `crossmatch_gaiaDR3`.
    clobber : bool
        Re-run the gaiaDR3 cross-match and overwrite an existing output file.
    outdir : str
        Directory to save ``Perren2023/TOIs_in_Perren2023_{match_method}.csv``
        to, and to read the ``gaiaDR3`` cross-match / ``Perren2023/UCC_members.parquet`` from.
    verbose : bool
        Log progress.

    Returns
    -------
    pd.DataFrame
        One row per TOI whose host star's Gaia DR3 id is a listed UCC member,
        with the UCC cluster columns joined on.
    """
    fp = Path(outdir, "Perren2023", f"TOIs_in_Perren2023_{match_method.value}.csv")
    if fp.exists() and not clobber:
        if verbose:
            logger.info(f"Loaded: {fp} (use --clobber to re-query and overwrite)")
        return pd.read_csv(fp, dtype={"GaiaDR3": "Int64"})

    matches = crossmatch_gaiaDR3(match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose)

    if verbose:
        logger.info(f"Loading Perren+2023 (UCC) cluster members from {outdir}/Perren2023")
    df_mem = load_cluster_members(catalog="Perren2023", data_path=outdir)
    df_mem = df_mem.rename(columns={"source_id": "GaiaDR3"})
    df_mem["GaiaDR3"] = df_mem["GaiaDR3"].astype("Int64")

    # Drop matches-side columns already present in df_mem (e.g. ra/dec/pmra/
    # pmdec/radial_velocity/parallax for match_method="id") so pandas doesn't
    # silently suffix them "_x"/"_y" on merge; the join key must survive this.
    keep_cols = [c for c in matches.columns if c == "GaiaDR3" or c not in df_mem.columns]
    final = pd.merge(df_mem, matches[keep_cols], on="GaiaDR3", how="inner")
    final = final.sort_values(by="TOI", ascending=False).reset_index(drop=True)

    fp.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(fp, index=False)
    if verbose:
        logger.info(f"Saved: {fp}")

    return final


def crossmatch_cluster_catalog(
    catalog: str,
    match_method: MatchMethod = MatchMethod.id,
    clobber: bool = False,
    outdir: str = DATA_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """TOI -> Vizier cluster-membership cross-match by Gaia DR3 source_id.

    Works for any two-table Vizier cluster catalog (cluster-level + member-
    level) whose member table has, or maps to, a Gaia DR3 ``source_id``
    column. Currently used for Hunt2023, Hunt2024, and CantatGaudin2020.

    Parameters
    ----------
    catalog : str
        Vizier catalog name, e.g. "Hunt2023" (must be a key in
        ``_CLUSTER_MEMBER_RENAME`` and resolvable by ``CatalogDownloader``).
    match_method : MatchMethod
        Which ``gaiaDR3`` cross-match (``id`` or ``position``) to join the
        catalog's member table onto; see `crossmatch_gaiaDR3`.
    clobber : bool
        Re-download the catalog and overwrite an existing output file.
    outdir : str
        Directory to save ``{catalog}/TOIs_in_{catalog}_{match_method}.csv``
        to, and to read the ``gaiaDR3`` cross-match from.
    verbose : bool
        Log progress.

    Returns
    -------
    pd.DataFrame
        One row per TOI whose host star's Gaia DR3 id is a listed member of
        this catalog, with the catalog's member columns joined on.
    """
    fp = Path(outdir, catalog, f"TOIs_in_{catalog}_{match_method.value}.csv")
    if fp.exists() and not clobber:
        if verbose:
            logger.info(f"Loaded: {fp} (use --clobber to re-query and overwrite)")
        return pd.read_csv(fp)

    if verbose:
        logger.info(f"Downloading {catalog} member table from Vizier")
    cd = CatalogDownloader(catalog_name=catalog, catalog_type="cluster", data_loc=outdir, verbose=verbose)
    tabs = cd.get_tables_from_vizier(row_limit=-1, save=True)
    df_mem = tabs[1].to_pandas().rename(columns=_CLUSTER_MEMBER_RENAME.get(catalog, {}))

    matches = crossmatch_gaiaDR3(match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose)
    final = pd.merge(df_mem, matches, on="GaiaDR3", how="inner")

    keep_cols = [c for c in _CLUSTER_OUTPUT_COLUMNS if c in final.columns]
    final = final[keep_cols].sort_values(by="TOI", ascending=False).reset_index(drop=True)

    fp.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(fp, index=False)
    if verbose:
        logger.info(f"Saved: {fp}")

    return final


def crossmatch_ratzenboeck2023(
    match_method: MatchMethod = MatchMethod.id,
    clobber: bool = False,
    outdir: str = DATA_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """TOI -> Ratzenboeck+2023 (Sco-Cen SigMA) cross-match by Gaia DR3 source_id.

    Ratzenboeck+2023 ships as two linked Vizier catalogs sharing a ``SigMA``
    group id: 2023a has per-star membership (Gaia source_id, SigMA id, no
    group name), 2023b has per-group names and ages. The exploratory notebook
    (``notebooks/crossmatch_TOI_with_Ratzenboeck2023.ipynb``) downloaded both
    but never joined them — its saved output has no cluster name or age at
    all. This joins member -> group name -> group age on ``SigMA`` to fix
    that before cross-matching against TOIs.

    Parameters
    ----------
    match_method : MatchMethod
        Which ``gaiaDR3`` cross-match (``id`` or ``position``) to join the
        SigMA member table onto; see `crossmatch_gaiaDR3`.
    clobber : bool
        Re-download the catalogs and overwrite an existing output file.
    outdir : str
        Directory to save
        ``Ratzenboeck2023/TOIs_in_Ratzenboeck2023_{match_method}.csv`` to,
        and to read the ``gaiaDR3`` cross-match from.
    verbose : bool
        Log progress.

    Returns
    -------
    pd.DataFrame
        One row per TOI whose host star's Gaia DR3 id is a listed SigMA
        member, with the group's name and age joined on.
    """
    fp = Path(outdir, "Ratzenboeck2023", f"TOIs_in_Ratzenboeck2023_{match_method.value}.csv")
    if fp.exists() and not clobber:
        if verbose:
            logger.info(f"Loaded: {fp} (use --clobber to re-query and overwrite)")
        return pd.read_csv(fp)

    if verbose:
        logger.info("Downloading Ratzenboeck2023a (membership) from Vizier")
    cd_a = CatalogDownloader(
        catalog_name="Ratzenboeck2023a", catalog_type="cluster", data_loc=outdir, verbose=verbose
    )
    tabs_a = cd_a.get_tables_from_vizier(row_limit=-1, save=True)
    df_groups = tabs_a[0].to_pandas()
    df_mem = tabs_a[1].to_pandas().rename(columns={"RV": "radial_velocity"})

    if verbose:
        logger.info("Downloading Ratzenboeck2023b (group ages) from Vizier")
    cd_b = CatalogDownloader(
        catalog_name="Ratzenboeck2023b", catalog_type="cluster", data_loc=outdir, verbose=verbose
    )
    tabs_b = cd_b.get_tables_from_vizier(row_limit=-1, save=True)
    df_ages = tabs_b[0].to_pandas()
    age_cols = [c for c in df_ages.columns if c.startswith(("Age", "e_Age", "E_Age"))]

    df_mem = df_mem.merge(df_groups[["SigMA", "Group"]], on="SigMA", how="left")
    df_mem = df_mem.merge(df_ages[["SigMA", *age_cols]], on="SigMA", how="left")
    df_mem = df_mem.rename(columns={"Group": "Name"})

    matches = crossmatch_gaiaDR3(match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose)
    final = pd.merge(df_mem, matches, on="GaiaDR3", how="inner")

    keep_cols = [c for c in [*_CLUSTER_OUTPUT_COLUMNS, *age_cols] if c in final.columns]
    final = final[keep_cols].sort_values(by="TOI", ascending=False).reset_index(drop=True)

    fp.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(fp, index=False)
    if verbose:
        logger.info(f"Saved: {fp}")

    return final


#: Catalogs eligible for --merge. gaiaDR3 is excluded: it's the underlying
#: TOI -> Gaia source_id lookup the others are built on, not itself a
#: cluster-membership statement, so it has no "Name"/cluster column to merge.
MERGEABLE_CATALOGS = [
    Catalog.Perren2023,
    Catalog.Hunt2023,
    Catalog.Hunt2024,
    Catalog.CantatGaudin2020,
    Catalog.Ratzenboeck2023,
]


def _run_catalog(
    catalog: Catalog, match_method: MatchMethod, clobber: bool, outdir: str, verbose: bool
) -> pd.DataFrame:
    """Dispatch to the right crossmatch_* function for a cluster catalog."""
    if catalog is Catalog.Perren2023:
        return crossmatch_perren2023(match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose)
    if catalog is Catalog.Ratzenboeck2023:
        return crossmatch_ratzenboeck2023(
            match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose
        )
    return crossmatch_cluster_catalog(
        catalog.value, match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose
    )


def merge_cluster_catalogs(
    match_method: MatchMethod = MatchMethod.id,
    clobber: bool = False,
    outdir: str = DATA_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cross-match TOIs against every cluster catalog and stack the results.

    Runs (or loads the cached output of) each catalog in
    ``MERGEABLE_CATALOGS`` and concatenates them, tagged with a ``Catalog``
    column identifying the source. A TOI matched by more than one catalog
    appears once per catalog rather than being deduplicated away, since each
    catalog's independent membership call is useful signal on its own.
    Catalogs have different columns (e.g. only Ratzenboeck2023 has
    ``AgeP-BpRp``); missing columns for a given catalog are filled with NaN
    rather than dropped.

    Parameters
    ----------
    match_method : MatchMethod
        Which ``gaiaDR3`` cross-match (``id`` or ``position``) every catalog
        joins onto; see `crossmatch_gaiaDR3`.
    clobber : bool
        Re-run every catalog cross-match and overwrite existing outputs.
    outdir : str
        Directory each per-catalog output is read from/saved to, and where
        the merged file is saved.
    verbose : bool
        Log progress.

    Returns
    -------
    pd.DataFrame
        Concatenated TOI-cluster matches from all catalogs in
        ``MERGEABLE_CATALOGS``, sorted by TOI then Catalog.
    """
    fp = Path(outdir, f"TOIs_in_clusters_merged_{match_method.value}.csv")
    if fp.exists() and not clobber:
        if verbose:
            logger.info(f"Loaded: {fp} (use --clobber to re-query and overwrite)")
        return pd.read_csv(fp, dtype={"GaiaDR3": "Int64"})

    frames = []
    for catalog in MERGEABLE_CATALOGS:
        df = _run_catalog(catalog, match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose).copy()
        df["Catalog"] = catalog.value
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    lead_cols = [c for c in ["TOI", "TIC ID", "Name", "GaiaDR3", "Prob", "Catalog"] if c in merged.columns]
    merged = merged[[*lead_cols, *[c for c in merged.columns if c not in lead_cols]]]
    merged = merged.sort_values(by=["TOI", "Catalog"]).reset_index(drop=True)

    fp.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(fp, index=False)
    if verbose:
        logger.info(f"Saved: {fp}")

    return merged


@app.command()
def main(
    catalog: Catalog = typer.Option(
        Catalog.gaiaDR3, "--catalog", help="External catalog to cross-match TOIs against."
    ),
    match_method: MatchMethod = typer.Option(
        MatchMethod.id,
        "--match_method",
        help="Which gaiaDR3 cross-match to use, i.e. which of "
        "TOIs_GaiaDR3_xmatch_id.csv / TOIs_GaiaDR3_xmatch_position.csv to "
        "read from (or write to, for --catalog gaiaDR3 itself): 'id' "
        "(default) via the TIC catalog's Gaia cross-ID, or 'position' via a "
        "sky-position cross-match with proper-motion epoch propagation. "
        "Applies to every --catalog and to --merge.",
    ),
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Ignore --catalog; cross-match against every cluster catalog and merge the results into one file.",
    ),
    outdir: str = typer.Option(DATA_PATH, help="Directory to save the cross-match CSV to."),
    clobber: bool = typer.Option(False, help="Re-download/re-query and overwrite existing output."),
    verbose: bool = typer.Option(True, help="Print progress messages."),
) -> None:
    """Cross-match the TOI catalog against an external catalog by TIC ID."""
    if merge:
        matches = merge_cluster_catalogs(match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose)
        n_tois = matches["TOI"].nunique() if "TOI" in matches.columns else 0
        typer.echo(
            f"Merged {len(matches)} TOI-cluster matches from {len(MERGEABLE_CATALOGS)} catalogs "
            f"via {match_method.value} matching ({n_tois} unique TOIs)."
        )
        return

    if catalog is Catalog.gaiaDR3:
        matches = crossmatch_gaiaDR3(match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose)
        n_with_gaia = int(matches["GaiaDR3"].notna().sum())
        typer.echo(
            f"Cross-matched {len(matches)} TOI rows against gaiaDR3 via {match_method.value} matching "
            f"({n_with_gaia} with a Gaia ID)."
        )
        return

    matches = _run_catalog(catalog, match_method=match_method, clobber=clobber, outdir=outdir, verbose=verbose)
    n_clusters = matches["Name"].nunique() if "Name" in matches.columns else 0
    typer.echo(
        f"Cross-matched {len(matches)} TOI rows against {catalog.value} via {match_method.value} matching "
        f"(members in {n_clusters} clusters)."
    )


if __name__ == "__main__":
    app()
