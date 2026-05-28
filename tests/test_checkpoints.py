import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from classifier.train import load_checkpoint_into_model, save_head_checkpoint


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 3
        self.classifier = nn.Linear(self.hidden, 1)


class CheckpointTest(unittest.TestCase):
    def test_head_checkpoint_round_trips_without_full_esm_state(self):
        source = DummyModel()
        with torch.no_grad():
            source.classifier.weight.fill_(0.25)
            source.classifier.bias.fill_(0.75)

        target = DummyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best_head.pt"
            save_head_checkpoint(source, checkpoint_path, metadata={"selected_val_threshold": 0.42})
            metadata = load_checkpoint_into_model(target, checkpoint_path)

        self.assertEqual(metadata["format"], "cruci_esm_classifier_head")
        self.assertEqual(metadata["metadata"]["selected_val_threshold"], 0.42)
        self.assertTrue(torch.equal(source.classifier.weight, target.classifier.weight))
        self.assertTrue(torch.equal(source.classifier.bias, target.classifier.bias))


if __name__ == "__main__":
    unittest.main()
