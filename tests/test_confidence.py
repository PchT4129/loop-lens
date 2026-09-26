import unittest

import torch

from src.confidence import apply_gate, fit_gate


class ConfidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.signals = {
            "retrieval_score": torch.tensor([0.95, 0.90, 0.40, 0.30]),
            "sequence_score": torch.tensor([0.95, 0.85, 0.45, 0.20]),
            "inlier_ratio": torch.tensor([0.40, 0.03, 0.30, 0.02]),
            "log_num_matches": torch.log1p(torch.tensor([100.0, 80.0, 60.0, 40.0])),
        }
        self.top1_correct = torch.tensor([True, False, True, False])
        self.has_match = torch.tensor([True, False, True, False])

    def test_joint_gate_requires_both_signals(self):
        config, metrics = fit_gate(
            "joint", self.signals, self.top1_correct, self.has_match
        )
        accepted = apply_gate(config, self.signals)

        self.assertEqual(accepted.tolist(), [True, False, True, False])
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_similarity_config_round_trips(self):
        config, _ = fit_gate(
            "similarity", self.signals, self.top1_correct, self.has_match
        )
        accepted = apply_gate(config, self.signals)
        self.assertEqual(accepted.dtype, torch.bool)
        self.assertEqual(len(accepted), 4)

    def test_logistic_config_runs_without_serialized_estimator(self):
        config, _ = fit_gate(
            "logistic", self.signals, self.top1_correct, self.has_match
        )
        accepted = apply_gate(config, self.signals)
        self.assertEqual(config["feature_names"][0], "retrieval_score")
        self.assertEqual(len(accepted), 4)


if __name__ == "__main__":
    unittest.main()
