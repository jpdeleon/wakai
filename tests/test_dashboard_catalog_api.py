import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from dashboard.backend.server import app


class DashboardCatalogApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        app.config["DATABASE_PATH"] = str(Path(self.tempdir.name) / "api.sqlite3")
        self.client = app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("dashboard.backend.server.query_vizier_reference")
    def test_vizier_endpoint_returns_reference_candidates(self, query_reference):
        query_reference.return_value = {
            "ref_table": [{"parameter": "Teff", "value": 5772, "id": "teff"}],
            "candidates": [{"parameter": "Teff", "value": 5772, "id": "teff"}],
            "available_parameters": ["Teff"],
            "tables": [{"catalog": "J/example/1"}],
        }

        response = self.client.post("/api/query/vizier", json={
            "name": "Sun",
            "ra": 0,
            "dec": 0,
            "radius": 3,
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["candidates"][0]["value"], 5772)
        query_reference.assert_called_once()

    @patch("dashboard.backend.server.simbad_target_payload")
    def test_simbad_endpoint_returns_resolved_target(self, resolve):
        resolve.return_value = {
            "name": "HD 1",
            "ra": 10.5,
            "dec": -2.25,
            "gaiaid": "123456",
            "source_url": "https://simbad.example/HD1",
        }

        response = self.client.post("/api/query/simbad", json={"name": "HD 1"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["target"]["gaiaid"], "123456")

    def test_vizier_endpoint_validates_radius(self):
        response = self.client.post("/api/query/vizier", json={
            "name": "Sun",
            "ra": 0,
            "dec": 0,
            "radius": 100,
        })

        self.assertEqual(response.status_code, 400)

    @patch("dashboard.backend.server._gaia_tap_query")
    def test_gaia_endpoint_preserves_catalog_precision(self, tap_query):
        tap_query.return_value = [[
            123456,
            10.123456789012,
            -2.987654321098,
            3.1415926535,
            0.0123456789,
            -12.345678901,
            5772.123456,
            5750.123456,
            5798.987654,
        ]]

        response = self.client.post("/api/query/gaia", json={"gaia_id": "123456"})

        self.assertEqual(response.status_code, 200)
        target = response.get_json()["target"]
        self.assertEqual(target["ra"], 10.123456789012)
        self.assertEqual(target["parallax"], 3.1415926535)
        self.assertEqual(target["parameters"][0]["value"], 5772.123456)

    def test_target_snapshot_round_trip(self):
        payload = {
            "target": {"name": "HD 1", "gaiaid": "123456", "ra": 10.123456789},
            "ref_table": [{
                "id": "teff",
                "parameter": "Teff",
                "value": 5772.123456,
                "uncertainty": 12.3456,
            }],
            "results": {"gyro": {"median": 456.789, "agePDF": [0.1, 0.2]}},
        }

        save_response = self.client.post("/api/targets/save", json=payload)
        load_response = self.client.get("/api/targets/load?gaia_id=123456")
        list_response = self.client.get("/api/targets")

        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(load_response.status_code, 200)
        record = load_response.get_json()["record"]
        self.assertEqual(record["ref_table"][0]["value"], 5772.123456)
        self.assertEqual(record["results"]["gyro"]["median"], 456.789)
        self.assertEqual(list_response.get_json()["targets"][0]["gaia_id"], "123456")

    def test_index_starts_without_a_reference_table_or_default_target(self):
        project_root = Path(__file__).resolve().parents[1]
        html = (project_root / "dashboard" / "index.html").read_text()
        javascript = (project_root / "dashboard" / "app.js").read_text()

        self.assertIn('id="ref-table-card" hidden', html)
        self.assertNotIn("loadTarget(state.preloadedTargets[0])", javascript)

    @patch("dashboard.backend.server._run_wrapper")
    def test_gyro_endpoint_returns_real_wrapper_posterior(self, run_wrapper):
        run_wrapper.return_value = {
            "status": "ok",
            "age_grid": [100, 200, 300],
            "pdf": [0.1, 0.8, 0.1],
            "median": 200,
            "lower_1sig": 150,
            "upper_1sig": 250,
        }

        response = self.client.post("/api/run/gyro", json={"teff": 5200, "prot": 8})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["median"], 200)
        self.assertEqual(run_wrapper.call_args.args[2]["prot"], 8)

    def test_gyro_endpoint_rejects_uncalibrated_temperature(self):
        response = self.client.post("/api/run/gyro", json={"teff": 7000, "prot": 8})

        self.assertEqual(response.status_code, 422)
        self.assertIn("outside the range", response.get_json()["message"])

    @patch("dashboard.backend.server._run_wrapper")
    def test_baffles_endpoint_derives_bv_then_runs_model(self, run_wrapper):
        run_wrapper.side_effect = [
            {"status": "ok", "bv": 0.7, "bv_err": 0.03},
            {
                "status": "ok",
                "age_grid": [10, 100, 1000],
                "ca_pdf": [0.1, 0.8, 0.1],
                "li_pdf": [0.2, 0.7, 0.1],
                "combined_pdf": [0.05, 0.9, 0.05],
                "ca_stats": [10, 50, 100, 200, 500],
                "li_stats": [10, 40, 90, 180, 450],
                "combined_stats": [20, 60, 95, 160, 300],
            },
        ]

        response = self.client.post(
            "/api/run/baffles",
            json={"teff": 5200, "rhk": -4.4, "liew": 120},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["bv_lookup"]["bv"], 0.7)
        self.assertEqual(run_wrapper.call_args_list[1].args[2]["bv"], 0.7)


if __name__ == "__main__":
    unittest.main()
