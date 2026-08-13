import tempfile
import unittest
from pathlib import Path

from wakai.database import TargetDatabase, target_key


class TargetDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = TargetDatabase(Path(self.tempdir.name) / "targets.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_save_and_load_target_snapshot(self):
        target = {
            "name": "HD 1",
            "gaiaid": "123456",
            "ra": 10.123456789,
            "dec": -2.987654321,
            "parallax": 3.14159265,
            "rv": -12.345678,
        }
        ref_table = [{
            "id": "teff",
            "parameter": "Teff",
            "value": 5772.123456,
            "uncertainty": 12.3456,
            "reference": "Example catalog",
            "catalog": "J/example/1",
            "column": "Teff",
            "unit": "K",
            "source_url": "https://example.test/catalog",
        }]
        results = {
            "gyro": {"median": 456.789, "agePDF": [0.1, 0.2, 0.3]},
            "joint": {"status": "Idle", "agePDF": None},
        }

        key = self.database.save(target, ref_table, results)
        record = self.database.load(gaia_id="123456")

        self.assertEqual(key, "gaia:123456")
        self.assertEqual(record["target"]["ra"], 10.123456789)
        self.assertEqual(record["ref_table"][0]["value"], 5772.123456)
        self.assertEqual(record["results"]["gyro"]["median"], 456.789)

    def test_save_replaces_previous_snapshot_for_same_target(self):
        target = {"name": "HD 1", "gaiaid": "123456"}
        self.database.save(target, [], {"gyro": {"median": 100}})
        self.database.save(target, [], {"gyro": {"median": 200}})

        record = self.database.load(gaia_id="123456")

        self.assertEqual(record["results"]["gyro"]["median"], 200)
        self.assertEqual(len(self.database.list_targets()), 1)

    def test_target_key_falls_back_to_normalized_name(self):
        self.assertEqual(target_key({"name": "TOI  123.01"}), "name:toi-123-01")


if __name__ == "__main__":
    unittest.main()
