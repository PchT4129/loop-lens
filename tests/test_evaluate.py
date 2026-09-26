import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

from src.evaluate import (
    apply_geometric_verification,
    bootstrap_open_set_cis,
    bootstrap_retrieval_cis,
    evaluate_retrieval,
    open_set_metrics,
    write_proposals,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.query_paths = [f"/query/night_right/Image{i:03d}.jpg" for i in range(3)]
        self.database_paths = [f"/database/day_left/Image{i:03d}.jpg" for i in range(3)]
        self.top_indices = torch.tensor([[0, 1], [2, 1], [2, 1]])

    def test_retrieval_metrics(self):
        metrics = evaluate_retrieval(
            self.query_paths, self.database_paths, self.top_indices,
            recall_ks=[1, 2], precision_k=2, tolerance=0,
        )
        self.assertAlmostEqual(metrics["recall@1"], 2 / 3)
        self.assertEqual(metrics["recall@2"], 1.0)

    def test_open_set_acceptance_is_explicit(self):
        metrics = open_set_metrics(
            self.query_paths,
            self.database_paths[:2],
            torch.tensor([[0], [1], [1]]),
            torch.ones(3),
            tolerance=0,
            accepted=torch.tensor([True, True, False]),
        )
        self.assertEqual(metrics["open_set/TP"], 2)
        self.assertEqual(metrics["open_set/TN"], 1)
        self.assertEqual(metrics["open_set/FP"], 0)

    def test_bootstrap_ci_contains_point_estimate(self):
        metrics = bootstrap_retrieval_cis(
            self.query_paths, self.database_paths, self.top_indices,
            recall_ks=[1], tolerance=0, samples=500, seed=7,
        )
        self.assertLessEqual(metrics["recall@1/ci95_low"], 2 / 3)
        self.assertGreaterEqual(metrics["recall@1/ci95_high"], 2 / 3)

    def test_open_set_bootstrap_ci_contains_point_estimate(self):
        metrics = bootstrap_open_set_cis(
            accepted=torch.tensor([True, True, False, False]),
            top1_correct=torch.tensor([True, True, False, False]),
            has_match=torch.tensor([True, True, False, False]),
            samples=500,
            seed=4,
        )
        self.assertLessEqual(metrics["deployed/open_set/f1/ci95_low"], 1.0)
        self.assertEqual(metrics["deployed/open_set/f1/ci95_high"], 1.0)

    def test_gate_mode_does_not_rerank(self):
        fake_module = ModuleType("src.geometric_verification")
        fake_module.verify_candidates = lambda *_args, **_kwargs: [
            {"num_matches": 10, "num_inliers": 1, "inlier_ratio": 0.1,
             "fundamental_matrix": None},
            {"num_matches": 10, "num_inliers": 9, "inlier_ratio": 0.9,
             "fundamental_matrix": None},
        ]
        with patch.dict("sys.modules", {"src.geometric_verification": fake_module}):
            indices, _, stats = apply_geometric_verification(
                [self.query_paths[0]], self.database_paths[:2],
                torch.tensor([[0, 1]]), torch.tensor([[0.9, 0.8]]),
                mode="gate", verifier="orb",
            )
        self.assertEqual(indices.tolist(), [[0, 1]])
        self.assertEqual(stats[0]["num_inliers"], 1)

    def test_proposal_json_contains_decision_evidence(self):
        signals = {
            "retrieval_score": torch.tensor([0.8]),
            "sequence_score": torch.tensor([0.9]),
            "inlier_ratio": torch.tensor([0.2]),
            "log_num_matches": torch.tensor([1.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "proposals.json"
            write_proposals(
                output, self.query_paths[:1], self.database_paths,
                torch.tensor([[0]]), signals,
                [{"num_matches": 20, "num_inliers": 4, "inlier_ratio": 0.2,
                  "fundamental_matrix": None}],
                torch.tensor([True]),
            )
            proposal = json.loads(output.read_text())[0]
        self.assertTrue(proposal["accepted"])
        self.assertEqual(proposal["num_inliers"], 4)


if __name__ == "__main__":
    unittest.main()
