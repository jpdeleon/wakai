"""Backward-compatible facade for the refactored catalog modules.

New code should import from the focused modules directly. Existing notebooks can
continue to import the historical names from :mod:`wakai.catalog`.
"""

from .catalog_keys import *
from .cluster import (
    ClusterMatch,
    assess_cluster_membership,
    compute_membership_probability,
    crossmatch_target_to_clusters,
    find_cluster_by_gaiaid,
    get_cluster_3d_size,
    get_cluster_summary,
    get_transformed_coord,
    load_cluster_catalog,
    load_cluster_members,
    plot_rdp_pmrv,
    plot_xyz_3d,
    plot_xyz_uvw,
)
from .config import DATA_PATH
from .iso import get_absolute_gmag, get_mist_eep_table, plot_cmd, plot_hrd
from .planet import (
    Planet,
    get_ctois,
    get_nexsci_data,
    get_params_from_tfop,
    get_tic_id,
    get_tois,
    query_tfop_data,
)
from .plotting import load_plot_params
from .reference import DEFAULT_PARAMETER_SPECS, query_vizier_reference, simbad_target_payload
from .sed import get_bv_from_teff, get_mamajek_table, interpolate_mamajek_table
from .star import Star, Target
from .utils import flatten_list
from .vizier import CatalogDownloader, crossmatch_catalogs
