import math
import unittest

import numpy as np

from classifier.data_utils import clean_protein_sequence, label_from_identifier
from classifier.metrics import binary_metrics


class DataUtilsTest(unittest.TestCase):
    def test_clean_protein_sequence_when_stops_and_invalid_chars_makes_esm_safe(self):
        cleaned = clean_protein_sequence(" ac*d?e ", stop_action="remove", invalid_action="replace_x")

        self.assertEqual(cleaned.sequence, "ACDXE")
        self.assertEqual(cleaned.original_length, 6)
        self.assertEqual(cleaned.cleaned_length, 5)
        self.assertEqual(cleaned.stops_removed, 1)
        self.assertEqual(cleaned.invalid_replaced, 1)
        self.assertTrue(cleaned.changed)

    def test_clean_protein_sequence_when_stop_action_error_raises(self):
        with self.assertRaisesRegex(ValueError, "stop"):
            clean_protein_sequence("AC*DE", stop_action="error")

    def test_label_from_identifier_when_cruci_substring_returns_positive(self):
        self.assertEqual(label_from_identifier("Cruci_CruV_123"), 1)
        self.assertEqual(label_from_identifier("Ribo_ND_123"), 0)


class MetricsTest(unittest.TestCase):
    def test_binary_metrics_when_imbalanced_returns_counts_and_balanced_metrics(self):
        result = binary_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.7, 0.1]), thr=0.5)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["balanced_acc"], 0.5)
        self.assertIn("average_precision", result)
        self.assertIn("mcc", result)

    def test_binary_metrics_when_single_class_does_not_raise_for_auroc(self):
        result = binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]), thr=0.5)

        self.assertTrue(math.isnan(result["auroc"]))
        self.assertTrue(math.isnan(result["average_precision"]))
        self.assertEqual(result["negatives"], 2)
        self.assertEqual(result["positives"], 0)


if __name__ == "__main__":
    unittest.main()
