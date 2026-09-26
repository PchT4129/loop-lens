import tempfile
import unittest
from pathlib import Path

import torch

from src.retrieve import load_feature_file


class FeatureArtifactTests(unittest.TestCase):
    def test_loads_metadata_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "features.pt"
            torch.save({
                "features": torch.ones(2, 3),
                "paths": ["a.jpg", "b.jpg"],
                "meta": {"backbone": "test", "feature_dim": 3},
            }, artifact)
            features, paths, meta = load_feature_file(artifact, include_meta=True)
        self.assertEqual(tuple(features.shape), (2, 3))
        self.assertEqual(paths, ["a.jpg", "b.jpg"])
        self.assertEqual(meta["feature_dim"], 3)

    def test_rejects_feature_path_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "bad.pt"
            torch.save({"features": torch.ones(2, 3), "paths": ["a.jpg"]}, artifact)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_feature_file(artifact)


if __name__ == "__main__":
    unittest.main()
