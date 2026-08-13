"""Unit tests for the wakai cluster membership methods.

These use small synthetic member tables so they run without touching the large
Perren+2023 data files.
"""
import unittest

import numpy as np
import pandas as pd

from wakai.cluster import (
    ClusterMatch,
    compute_membership_probability,
    find_cluster_by_gaiaid,
    get_cluster_summary,
)


def _tight_cluster(n=200, seed=0):
    """A compact synthetic cluster in (parallax, pmra, pmdec, rv)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1),
            "Name": ["mock1"] * n,
            "ra": rng.normal(150.0, 0.2, n),
            "dec": rng.normal(-30.0, 0.2, n),
            "parallax": rng.normal(5.0, 0.05, n),
            "pmra": rng.normal(10.0, 0.3, n),
            "pmdec": rng.normal(-4.0, 0.3, n),
            "radial_velocity": rng.normal(12.0, 0.5, n),
            "Prob": rng.uniform(0.5, 1.0, n),
        }
    )


class MembershipProbabilityTests(unittest.TestCase):
    def test_target_at_cluster_center_is_highly_probable(self):
        members = _tight_cluster()
        target = {"parallax": 5.0, "pmra": 10.0, "pmdec": -4.0, "radial_velocity": 12.0}

        result = compute_membership_probability(target, members)

        self.assertGreater(result["probability"], 0.5)
        self.assertEqual(result["dof"], 4)
        self.assertIn("radial_velocity", result["features"])

    def test_kinematic_outlier_is_rejected(self):
        members = _tight_cluster()
        target = {"parallax": 9.0, "pmra": 40.0, "pmdec": 30.0, "radial_velocity": -50.0}

        result = compute_membership_probability(target, members)

        self.assertLess(result["probability"], 1e-3)
        self.assertGreater(result["mahalanobis"], 5)

    def test_radial_velocity_dropped_when_target_lacks_it(self):
        members = _tight_cluster()
        target = {"parallax": 5.0, "pmra": 10.0, "pmdec": -4.0}  # no RV

        result = compute_membership_probability(target, members)

        self.assertEqual(result["dof"], 3)
        self.assertNotIn("radial_velocity", result["features"])

    def test_returns_empty_when_too_few_usable_features(self):
        members = _tight_cluster()
        target = {"parallax": 5.0}  # only one feature -> cannot form a covariance

        result = compute_membership_probability(target, members)

        self.assertIsNone(result["probability"])
        self.assertEqual(result["dof"], 0)


class FindClusterByGaiaIdTests(unittest.TestCase):
    def test_match_returns_only_the_matching_rows(self):
        members = _tight_cluster(n=5)
        members.loc[2, "source_id"] = 424242

        hits = find_cluster_by_gaiaid(424242, members=members)

        self.assertEqual(len(hits), 1)
        self.assertEqual(int(hits.iloc[0]["source_id"]), 424242)

    def test_absent_id_returns_empty_frame(self):
        members = _tight_cluster(n=5)

        hits = find_cluster_by_gaiaid(999999999, members=members)

        self.assertTrue(hits.empty)


class ClusterSummaryTests(unittest.TestCase):
    def test_summary_labels_and_filters_missing_columns(self):
        catalog = pd.DataFrame(
            {
                "Name": ["mock1"],
                "Name(s)": ["Mock 1"],
                "Age_[Myr]": [120.0],
                "Dist_[kpc]": [0.3],
                "FeH_[dex]": [np.nan],
            }
        )

        summary = get_cluster_summary("mock1", catalog_df=catalog)
        labels = {item["label"]: item for item in summary}

        self.assertEqual(labels["Age"]["value"], 120.0)
        self.assertEqual(labels["Age"]["unit"], "Myr")
        # A NaN metallicity is normalized to None rather than a non-JSON float.
        self.assertIsNone(labels["[Fe/H]"]["value"])

    def test_unknown_cluster_returns_empty_summary(self):
        catalog = pd.DataFrame({"Name": ["mock1"], "Age_[Myr]": [10.0]})

        self.assertEqual(get_cluster_summary("nope", catalog_df=catalog), [])


class ClusterMatchDataclassTests(unittest.TestCase):
    def test_dataclass_holds_expected_fields(self):
        match = ClusterMatch(
            matched=True,
            method="gaia_id",
            cluster_name="mock1",
            membership_probability=0.8,
            catalog_probability=0.8,
            mahalanobis=None,
            dof=None,
            sep3d_pc=5.0,
            extent_pc=50.0,
            within_extent=True,
            n_members=200,
        )

        self.assertTrue(match.matched)
        self.assertEqual(match.method, "gaia_id")
        self.assertTrue(match.members.empty)


if __name__ == "__main__":
    unittest.main()
