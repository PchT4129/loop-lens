import unittest

import torch

from src.sequence_match import sequence_rerank


class SequenceMatchTests(unittest.TestCase):
    def test_causal_score_cannot_see_future_rows(self):
        similarities = torch.zeros(5, 5)
        baseline = sequence_rerank(similarities, window=2, causal=True)

        changed = similarities.clone()
        changed[4, :] = 100
        rescored = sequence_rerank(changed, window=2, causal=True)

        self.assertTrue(torch.equal(baseline[:4], rescored[:4]))

    def test_non_positive_window_is_identity(self):
        similarities = torch.rand(3, 4)
        self.assertIs(sequence_rerank(similarities, window=0), similarities)


if __name__ == "__main__":
    unittest.main()
