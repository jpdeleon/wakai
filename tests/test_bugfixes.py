"""Regression tests for bugs found during the package review (offline only)."""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from astropy.table import Table

from wakai.cluster import plot_xyz_3d, plot_xyz_uvw
from wakai.planet import Planet, get_tois
from wakai.star import Star

# ExoFOP groups parameter rows under a provenance header row that carries
# only "prov"; the data rows that follow it have no "prov" of their own.
TFOP_JSON = {
    "basic_info": {"tic_id": "150428135", "star_names": "TOI-700, Gaia DR3 5284517766615492736"},
    "coordinates": {"ra": "97.096", "dec": "-65.579"},
    "stellar_parameters": [
        {"prov": "tic"},
        {"srad": "0.42", "srad_e": "0.01", "mass": "0.40", "mass_e": "0.02",
         "teff": "3480", "teff_e": "157", "logg": "4.8", "logg_e": "0.1",
         "feh": "", "feh_e": "", "sdate": "2019-04-15"},
        {"prov": "toi"},
        {"srad": "0.99", "mass": "0.99", "teff": "9999", "sdate": "2024-02-15"},
    ],
    "planet_parameters": [
        {"prov": "confirmed"},
        {"name": "TOI-700 b", "per": "9.97", "epoch": "1.0", "dur": "2.1", "dep_p": "400"},
        {"prov": "toi"},
        {"toi": "TOI 700.03", "per": "9.977", "per_e": "0.0001", "epoch": "2458331.35",
         "epoch_e": "0.002", "dur": "2.4", "dur_e": "0.4", "dep_p": "459.85", "dep_p_e": "35.2"},
        {"toi": "TOI 700.01", "per": "16.05", "per_e": "0.0001", "epoch": "2458332.0",
         "epoch_e": "0.002", "dur": "1.2", "dur_e": "0.05", "dep_p": "900", "dep_p_e": "40"},
    ],
}


def _cluster_frame(n=20):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "source_id": np.arange(n) + 1,
        "ra": 100 + rng.normal(0, 1, n),
        "dec": -20 + rng.normal(0, 1, n),
        "parallax": 10 + rng.normal(0, 0.2, n),
        "pmra": 5 + rng.normal(0, 0.5, n),
        "pmdec": -3 + rng.normal(0, 0.5, n),
        "radial_velocity": 20 + rng.normal(0, 1, n),
        "teff_val": rng.uniform(4000, 6000, n),
    })


class StarParameterTests(unittest.TestCase):
    def test_star_params_skip_provenance_header_row(self):
        star = Star("TOI-700", tfop_data=TFOP_JSON, data_json=TFOP_JSON, verbose=False)
        star.get_star_params()
        self.assertAlmostEqual(star.rstar[0], 0.42)
        self.assertAlmostEqual(star.teff[0], 3480)

    def test_toiid_is_int_for_dashed_names(self):
        star = Star("TOI-700", tfop_data=TFOP_JSON, verbose=False)
        star.get_tfop_data()
        self.assertEqual(star.toiid, 700)

    def test_gaia_metadata_does_not_need_optional_columns(self):
        star = Star("X", ra_deg=1.0, dec_deg=1.0, verbose=False)
        with patch.object(Star, "get_vizier_param", return_value={}):
            star._assign_gaia_metadata(pd.Series({"source_id": 5}))
        self.assertEqual(star.gaiaid, 5)


class GaiaCacheTests(unittest.TestCase):
    def _fake_gaia(self, n):
        return Table({
            "source_id": np.arange(n) + 1,
            "distance": np.linspace(0.001, 0.01, n),
            "parallax": np.ones(n),
            "ref_epoch": np.full(n, 2015.5),
        })

    def test_cache_respects_radius_and_nearest_flag(self):
        star = Star("X", ra_deg=1.0, dec_deg=1.0, verbose=False)
        with patch("wakai.star.Catalogs.query_region", side_effect=[self._fake_gaia(1), self._fake_gaia(5)]) as q, \
                patch.object(Star, "get_vizier_param", return_value={}):
            small = star.query_gaia_catalog(radius=3)
            big = star.query_gaia_catalog(radius=30)
            nearest = star.query_gaia_catalog(radius=30, return_nearest_xmatch=True)
        self.assertEqual(len(small), 1)
        self.assertEqual(len(big), 5)
        self.assertIsInstance(nearest, pd.Series)
        self.assertEqual(q.call_count, 2)


class VizierCacheTests(unittest.TestCase):
    def test_cache_is_refreshed_when_radius_changes(self):
        star = Star("X", ra_deg=1.0, dec_deg=1.0, verbose=False)
        # wakai.star.Vizier is an instance; Vizier(columns=...) returns a fresh
        # VizierClass, so the method must be patched on the class.
        with patch("astroquery.vizier.VizierClass.query_region", side_effect=[["a"], ["b"]]) as q:
            first = star.query_vizier(radius=3, verbose=False)
            again = star.query_vizier(verbose=False)  # default radius is 3"
            wider = star.query_vizier(radius=60, verbose=False)
        self.assertEqual(first, ["a"])
        self.assertEqual(again, ["a"])
        self.assertEqual(wider, ["b"])
        self.assertEqual(q.call_count, 2)


class PlanetTests(unittest.TestCase):
    def _planet(self, alias):
        with patch.object(Star, "query_tfop_data", return_value=TFOP_JSON):
            return Planet("TOI-700", alias=alias, tfop_data=TFOP_JSON, data_json=TFOP_JSON, verbose=False)

    def test_alias_selects_matching_toi_row(self):
        self.assertAlmostEqual(self._planet("01").period[0], 16.05)
        self.assertAlmostEqual(self._planet("03").period[0], 9.977)

    def test_duration_uses_dur_key_in_days(self):
        planet = self._planet("03")
        self.assertAlmostEqual(planet.tdur[0], 2.4 / 24)


class GetToisTests(unittest.TestCase):
    def test_remove_known_planets_keeps_rows_without_comments(self, tmp=None):
        import tempfile
        df = pd.DataFrame({
            "TOI": np.arange(1, 1202) + 0.01,
            "TFOPWG Disposition": "PC",
            "Comments": [np.nan] * 1200 + ["WASP-12 b"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            df.to_csv(f"{tmp}/TOIs.csv", index=False)
            out = get_tois(outdir=tmp, remove_known_planets=True)
        self.assertEqual(len(out), 1200)


class ClusterPlotTests(unittest.TestCase):
    def test_plot_xyz_3d_plots_all_points_without_limits(self):
        df = _cluster_frame()
        fig = plot_xyz_3d(df.copy())
        offsets = fig.axes[0].collections[0]._offsets3d
        self.assertEqual(len(offsets[0]), len(df))

    def test_plot_xyz_uvw_does_not_mutate_input(self):
        df = _cluster_frame()
        df[["X", "Y", "Z"]] = 5000.0
        df[["U", "V", "W"]] = 1.0
        before = df.copy()
        plot_xyz_uvw(df, verbose=False)
        pd.testing.assert_frame_equal(df, before)


if __name__ == "__main__":
    unittest.main()
