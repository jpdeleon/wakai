import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from gui.backend.server import app


class GuiCatalogApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        app.config["DATABASE_PATH"] = str(Path(self.tempdir.name) / "api.sqlite3")
        self.client = app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    @patch("gui.backend.server.query_vizier_reference")
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

    @patch("gui.backend.server.simbad_target_payload")
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

    @patch("gui.backend.server._gaia_tap_query")
    def test_gaia_endpoint_preserves_catalog_precision(self, tap_query):
        # query_gaia issues two ADQL queries: the astrometry/Teff SELECT
        # (source_id, ra, dec, parallax, parallax_error, pmra, pmra_error,
        # pmdec, pmdec_error, radial_velocity, radial_velocity_error,
        # teff_gspphot, teff_gspphot_lower, teff_gspphot_upper), then
        # _fetch_gaia_2mass_photometry's join (parallax, parallax_error,
        # BP mag/flux_over_error, G mag/flux_over_error, RP mag/flux_over_error,
        # J/H/K mag+err) -- mocked in that call order via side_effect.
        tap_query.side_effect = [
            [[
                123456,
                10.123456789012,
                -2.987654321098,
                3.1415926535,
                0.0123456789,
                -12.345678901,
                0.045678901,
                -8.765432109,
                0.034567890,
                25.432109876,
                0.567891234,
                5772.123456,
                5750.123456,
                5798.987654,
            ]],
            [[
                3.1415926535, 0.0123456789,
                12.909325, 200.5,
                12.317167, 800.1,
                11.58319, 300.7,
                10.629, 0.022,
                10.12, 0.027,
                9.976, 0.024,
            ]],
        ]

        response = self.client.post("/api/query/gaia", json={"gaia_id": "123456"})

        self.assertEqual(response.status_code, 200)
        target = response.get_json()["target"]
        self.assertEqual(target["ra"], 10.123456789012)
        self.assertEqual(target["parallax"], 3.1415926535)
        parameters = {p["id"]: p for p in target["parameters"]}
        self.assertEqual(parameters["teff"]["value"], 5772.123456)
        self.assertEqual(parameters["parallax"]["value"], 3.1415926535)
        self.assertEqual(parameters["pmra"]["uncertainty"], 0.045678901)
        self.assertEqual(parameters["jmag"]["value"], 10.629)

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

    def test_page_settings_round_trip_as_an_ordinary_results_key(self):
        # gui/app.js persists per-page UI controls (gyro axis limits, cluster
        # catalog/name override, isochrones checkboxes/logg/priors) by
        # piggybacking on results.pageSettings -- relies on model_results
        # being a generic {method: json_payload} store with no schema
        # awareness of what "pageSettings" means. This locks that in.
        page_settings = {
            "gyro": {"showModels": True, "modelAges": [80, 300], "xmin": 3400},
            "cluster": {"catalog": "Perren2023", "clusterName": "IC 2602"},
            "isochrones": {
                "checkedIds": ["parallax", "bpmag"],
                "logg": {"checked": False, "value": 4.4, "err": 0.15},
                "agePrior": [6, 9],
            },
        }
        payload = {
            "target": {"name": "HD 2", "gaiaid": "654321"},
            "ref_table": [],
            "results": {"gyro": {"status": "Idle"}, "pageSettings": page_settings},
        }

        save_response = self.client.post("/api/targets/save", json=payload)
        load_response = self.client.get("/api/targets/load?gaia_id=654321")

        self.assertEqual(save_response.status_code, 200)
        self.assertEqual(load_response.get_json()["record"]["results"]["pageSettings"], page_settings)

    def test_index_starts_without_a_reference_table_or_default_target(self):
        project_root = Path(__file__).resolve().parents[1]
        html = (project_root / "gui" / "index.html").read_text()
        javascript = (project_root / "gui" / "app.js").read_text()

        self.assertIn('id="ref-table-card" hidden', html)
        self.assertNotIn("preloadedTargets", javascript)

    @patch("gui.backend.server._run_wrapper")
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

    @patch("gui.backend.server._run_wrapper")
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

    def test_isochrones_endpoint_requires_photometry_and_parallax(self):
        response = self.client.post("/api/run/isochrones", json={
            "parallax": 6.6, "parallax_err": 0.01,
        })

        self.assertEqual(response.status_code, 400)

    @patch("gui.backend.server._run_wrapper")
    def test_isochrones_endpoint_forwards_checked_inputs_only(self, run_wrapper):
        # Mirrors what the frontend now sends: photometry/parallax read
        # straight from ref_table (not re-fetched by gaiaid), plus whichever
        # optional teff/feh/logg the user has checked on the page.
        run_wrapper.return_value = {"status": "ok", "median": 500, "used_inputs": {}}

        response = self.client.post("/api/run/isochrones", json={
            "photometry": {"BP": [12.9, 0.003], "G": [12.3, 0.001]},
            "parallax": 6.6, "parallax_err": 0.01,
            "logg": 4.5, "logg_err": 0.1,
        })

        self.assertEqual(response.status_code, 200)
        sent = run_wrapper.call_args.args[2]
        self.assertEqual(sent["photometry"]["BP"], [12.9, 0.003])
        self.assertEqual(sent["logg"], 4.5)
        self.assertNotIn("teff", sent)
        self.assertNotIn("feh", sent)

    @patch("gui.backend.server._run_wrapper")
    def test_isochrones_endpoint_forwards_prior_bounds(self, run_wrapper):
        run_wrapper.return_value = {"status": "ok", "median": 500, "used_inputs": {}}

        response = self.client.post("/api/run/isochrones", json={
            "photometry": {"BP": [12.9, 0.003]},
            "parallax": 6.6, "parallax_err": 0.01,
            "age_prior_bounds": [6, 8], "av_prior_bounds": [0, 0.05],
            "eep_prior_bounds": [0, 500],
        })

        self.assertEqual(response.status_code, 200)
        sent = run_wrapper.call_args.args[2]
        self.assertEqual(sent["age_prior_bounds"], [6.0, 8.0])
        self.assertEqual(sent["av_prior_bounds"], [0.0, 0.05])
        self.assertEqual(sent["eep_prior_bounds"], [0.0, 500.0])

    def test_isochrones_endpoint_rejects_inverted_prior_bounds(self):
        response = self.client.post("/api/run/isochrones", json={
            "photometry": {"BP": [12.9, 0.003]},
            "parallax": 6.6, "parallax_err": 0.01,
            "age_prior_bounds": [10, 5],
        })

        self.assertEqual(response.status_code, 400)

    def test_madys_cmd_endpoint_requires_gaia_id(self):
        response = self.client.post("/api/run/madys-cmd", json={})

        self.assertEqual(response.status_code, 400)

    @patch("gui.backend.server._run_wrapper")
    def test_madys_cmd_endpoint_forwards_cluster_name_and_models(self, run_wrapper):
        run_wrapper.return_value = {"status": "ok", "plots": {"mist": "data:image/png;base64,AAA"}, "n_members": 42}

        response = self.client.post("/api/run/madys-cmd", json={
            "gaiaid": "5239758155778687360",
            "cluster_name": "ic2602",
            "models": ["mist", "parsec2"],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["n_members"], 42)
        sent = run_wrapper.call_args.args[2]
        self.assertEqual(sent["gaiaid"], "5239758155778687360")
        self.assertEqual(sent["cluster_name"], "ic2602")
        self.assertEqual(sent["models"], ["mist", "parsec2"])

    def test_madys_cmd_endpoint_rejects_empty_models_list(self):
        response = self.client.post("/api/run/madys-cmd", json={
            "gaiaid": "123456",
            "models": [],
        })

        self.assertEqual(response.status_code, 400)

    @patch("gui.backend.server._run_wrapper")
    def test_madys_cmd_endpoint_cluster_name_is_optional(self, run_wrapper):
        run_wrapper.return_value = {"status": "ok", "plot": "data:image/png;base64,AAA", "n_members": 0}

        response = self.client.post("/api/run/madys-cmd", json={"gaiaid": "123456"})

        self.assertEqual(response.status_code, 200)
        sent = run_wrapper.call_args.args[2]
        self.assertIsNone(sent["cluster_name"])

    @patch("gui.backend.server._run_wrapper")
    def test_madys_cmd_endpoint_forwards_name_and_plot_selections(self, run_wrapper):
        run_wrapper.return_value = {"status": "ok", "plot": "data:image/png;base64,AAA", "n_members": 0}

        response = self.client.post("/api/run/madys-cmd", json={
            "gaiaid": "123456",
            "name": "TOI-6715",
            "plot_ages": [10, 30],
            "plot_masses": [],
        })

        self.assertEqual(response.status_code, 200)
        sent = run_wrapper.call_args.args[2]
        self.assertEqual(sent["name"], "TOI-6715")
        self.assertEqual(sent["plot_ages"], [10, 30])
        self.assertEqual(sent["plot_masses"], [])


if __name__ == "__main__":
    unittest.main()
