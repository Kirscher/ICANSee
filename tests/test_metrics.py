import unittest
import numpy as np

from metrics import compute_metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        self.y_true = np.array([0, 0, 0, 1, 1])
        self.y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65])

    def test_compute_metrics_keys(self):
        metrics = compute_metrics(self.y_true, self.y_scores)
        expected_keys = [
            "AUROC",
            "AUPRC",
            "PPV@90% Recall",
            "Accuracy",
            "Sensitivity",
            "Specificity",
        ]
        self.assertEqual(set(metrics.keys()), set(expected_keys))

    def test_compute_metrics_values(self):
        metrics = compute_metrics(self.y_true, self.y_scores)
        # Check that all values are between 0 and 1
        for key, value in metrics.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.3, 0.4, 0.6, 0.7])
        metrics = compute_metrics(y_true, y_scores)
        self.assertEqual(metrics["Accuracy"], 1.0)
        self.assertEqual(metrics["Sensitivity"], 1.0)
        self.assertEqual(metrics["Specificity"], 1.0)

    def test_all_positive_class(self):
        y_true = np.array([1, 1, 1, 1])
        y_scores = np.array([0.5, 0.6, 0.7, 0.8])
        metrics = compute_metrics(y_true, y_scores)
        # AUROC undefined for single class
        self.assertTrue(np.isnan(metrics["AUROC"]) or metrics["AUROC"] == 0.5)
        self.assertEqual(metrics["Sensitivity"], 1.0)
        # Specificity undefined (no negative samples)
        self.assertEqual(metrics["Specificity"], 0.0)

    def test_all_negative_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = compute_metrics(y_true, y_scores)
        # AUROC undefined for single class
        self.assertTrue(np.isnan(metrics["AUROC"]) or metrics["AUROC"] == 0.5)
        # Sensitivity undefined (no positive samples)
        self.assertEqual(metrics["Sensitivity"], 0.0)
        self.assertEqual(metrics["Specificity"], 1.0)


if __name__ == "__main__":
    unittest.main()
