import unittest
from unittest.mock import patch

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table

import wakai.catalog as catalog
from wakai.cluster import get_cluster_3d_size
from wakai.iso import get_absolute_gmag
from wakai.planet import get_params_from_tfop
from wakai.reference import query_vizier_reference, simbad_target_payload
from wakai.star import Star
from wakai.vizier import CatalogDownloader, crossmatch_catalogs


class CatalogModuleTests(unittest.TestCase):
    def test_compatibility_facade_reexports_moved_symbols(self):
        self.assertIs(catalog.Star, Star)
        self.assertIs(catalog.CatalogDownloader, CatalogDownloader)
        self.assertEqual(catalog.Star.__module__, "wakai.star")
        self.assertEqual(catalog.plot_cmd.__module__, "wakai.iso")
        self.assertEqual(catalog.plot_xyz_uvw.__module__, "wakai.cluster")

    def test_star_can_be_created_without_a_network_query(self):
        star = Star("TIC 123", ra_deg=10.0, dec_deg=-20.0, verbose=False)

        self.assertEqual(star.target_coord().ra.deg, 10.0)
        self.assertEqual(star.target_coord().dec.deg, -20.0)
        self.assertIn("TIC 123", repr(star))

    def test_catalog_downloader_searches_every_registry(self):
        downloader = CatalogDownloader(catalog_name="Samus2017", verbose=False)

        self.assertEqual(downloader.vizier_key, "B/gcvs/gcvs_cat")

    def test_crossmatch_applies_optional_separation_limit(self):
        left = SkyCoord([0.0, 1.0] * u.deg, [0.0, 0.0] * u.deg)
        right = SkyCoord([0.0001] * u.deg, [0.0] * u.deg)

        indices, separations, mask = crossmatch_catalogs(left, right, max_sep_arcsec=1)

        self.assertEqual(indices.tolist(), [0, 0])
        self.assertLess(separations[0].arcsec, 1)
        self.assertEqual(mask.tolist(), [True, False])

    def test_cluster_size_and_absolute_magnitude_helpers(self):
        group = pd.DataFrame(
            {"ra": [0.0, 0.0], "dec": [0.0, 0.0], "parallax": [100.0, 50.0]}
        )

        self.assertAlmostEqual(get_cluster_3d_size(group), 10.0)
        self.assertAlmostEqual(get_absolute_gmag(10.0, 10.0, 0.0), 10.0)

    def test_latest_tfop_parameters_are_selected(self):
        data = {
            "stellar_parameters": [
                {"sdate": "2024-01-01", "teff": 5000},
                {"sdate": "2025-01-01", "teff": 5100},
            ]
        }

        latest = get_params_from_tfop(data, name="stellar_parameters")

        self.assertEqual(latest["teff"], 5100)

    def test_vizier_candidates_include_uncertainty_and_source_url(self):
        table = Table(
            {
                "Teff": [5772.0],
                "e_Teff": [42.0],
                "Prot": [25.4],
                "_r": [0.12],
            }
        )
        table.meta["description"] = "Example stellar catalog"

        class FakeTables:
            def __init__(self, key, value):
                self.mapping = {key: value}

            def __len__(self):
                return len(self.mapping)

            def keys(self):
                return list(self.mapping.keys())

            def values(self):
                return list(self.mapping.values())

            def __iter__(self):
                return iter(self.mapping.values())

        star = Star("Sun", ra_deg=0, dec_deg=0, verbose=False)
        star._vizier_tables = FakeTables("J/example/1", table)

        matches = star.query_vizier_param("Teff", detailed=True)
        result = query_vizier_reference(star)

        self.assertEqual(matches[0]["uncertainty"], 42.0)
        self.assertEqual(matches[0]["reference"], "Example stellar catalog")
        self.assertTrue(matches[0]["source_url"].startswith("https://vizier"))
        self.assertIn("-c=0%200", matches[0]["source_url"])
        self.assertEqual(result["ref_table"][0]["value"], 5772.0)
        self.assertEqual(result["ref_table"][1]["value"], 25.4)
        self.assertIsNone(result["ref_table"][2]["value"])

    def test_simbad_payload_extracts_coordinates_and_gaia_id(self):
        result = Table({
            "main_id": ["HD 1"],
            "ra": [10.5],
            "dec": [-2.25],
            "otype": ["Star"],
            "ids": ["HD 1|Gaia DR3 123456"],
        })
        star = Star("HD 1", verbose=False)

        with patch.object(star, "query_simbad", return_value=result):
            payload = simbad_target_payload(star)

        self.assertEqual(payload["gaiaid"], "123456")
        self.assertEqual(payload["ra"], 10.5)
        self.assertIn("simbad", payload["source_url"])


if __name__ == "__main__":
    unittest.main()
