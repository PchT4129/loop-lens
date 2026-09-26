import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluate import evaluate_retrieval, open_set_targets
from src.ground_truth import PairManifestGroundTruth


class PairManifestGroundTruthTests(unittest.TestCase):
    def test_distance_threshold_and_open_set_query(self):
        content = (
            "query_path,database_path,distance_m\n"
            "queries/q0.jpg,database/d0.jpg,4.5\n"
            "queries/q0.jpg,database/d1.jpg,31.0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "pairs.csv"
            manifest.write_text(content)
            ground_truth = PairManifestGroundTruth.from_csv(manifest, 25.0)

        self.assertTrue(
            ground_truth.is_correct("queries/q0.jpg", "database/d0.jpg")
        )
        self.assertFalse(
            ground_truth.is_correct("queries/q0.jpg", "database/d1.jpg")
        )
        self.assertFalse(
            ground_truth.has_match("queries/q1.jpg", ["database/d0.jpg"])
        )

        metrics = evaluate_retrieval(
            ["queries/q0.jpg", "queries/q1.jpg"],
            ["database/d0.jpg", "database/d1.jpg"],
            torch.tensor([[0], [1]]),
            recall_ks=[1],
            precision_k=1,
            tolerance=3,
            ground_truth=ground_truth,
        )
        _, has_match = open_set_targets(
            ["queries/q0.jpg", "queries/q1.jpg"],
            ["database/d0.jpg", "database/d1.jpg"],
            torch.tensor([[0], [1]]),
            tolerance=3,
            ground_truth=ground_truth,
        )
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(has_match.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
