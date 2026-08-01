from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ensemble.spe import (
    architecture_disagreement,
    fit_architecture_weights,
    patient_equal_sample_weights,
    residual_similarity_matrix,
)
from scripts.Diagnosis.run_spe import (
    architecture_test_prediction,
    trajectory_predictions,
)
from utils.spe_model_utils import _evenly_spaced_records, _stable_intervals


class TestSPE(unittest.TestCase):
    def test_patient_equal_weights(self):
        patients = np.array(["a", "a", "a", "b"])
        weights = patient_equal_sample_weights(patients)
        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3]), 0.5)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_architecture_fit_is_simplex_and_favors_accuracy(self):
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        patients = np.array([f"p{index}" for index in range(len(labels))])
        good = np.array([0.05, 0.15, 0.85, 0.95, 0.10, 0.90, 0.20, 0.80])
        weak = np.array([0.45, 0.55, 0.55, 0.45, 0.60, 0.40, 0.50, 0.50])
        complement = np.array([0.10, 0.25, 0.75, 0.90, 0.05, 0.80, 0.15, 0.95])
        matrix = np.column_stack([good, weak, complement])
        fit, similarity = fit_architecture_weights(
            matrix,
            labels,
            patients,
            architecture_names=["good", "weak", "complement"],
            diversity_lambda=0.05,
        )
        weights = np.asarray(fit.weights)
        self.assertTrue(fit.converged)
        self.assertTrue((weights >= 0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=8)
        self.assertGreater(weights[0] + weights[2], weights[1])
        self.assertTrue(np.allclose(similarity, similarity.T))
        self.assertTrue(np.allclose(np.diag(similarity), 1.0))

    def test_residual_similarity_and_disagreement(self):
        probabilities = np.array([[0.1, 0.1], [0.9, 0.7], [0.4, 0.6]])
        labels = np.array([0, 1, 1])
        sample_weights = np.full(3, 1 / 3)
        similarity = residual_similarity_matrix(
            probabilities, labels, sample_weights
        )
        self.assertEqual(similarity.shape, (2, 2))
        disagreement = architecture_disagreement(probabilities, [0.5, 0.5])
        np.testing.assert_allclose(disagreement, [0.0, 0.1, 0.1], atol=1e-8)

    def test_paper_stable_interval_and_even_spacing(self):
        values = [0.70, 0.701, 0.702, 0.701, 0.703, 0.720, 0.721, 0.722, 0.721, 0.720, 0.719]
        records = [
            {
                "epoch": index + 1,
                "checkpoint": f"epoch_{index + 1}.pth",
                "spe_metric_value": value,
            }
            for index, value in enumerate(values)
        ]
        intervals = _stable_intervals(records, threshold=0.003, min_consecutive=5)
        self.assertEqual([(run[0]["epoch"], run[-1]["epoch"]) for run in intervals], [(1, 5), (6, 11)])
        selected = _evenly_spaced_records(intervals[-1], count=5)
        self.assertEqual([record["epoch"] for record in selected], [6, 7, 8, 10, 11])

    def test_three_level_probability_aggregation(self):
        metadata = pd.DataFrame(
            {
                "slide_id": ["s1", "s2"],
                "patient_id": ["p1", "p2"],
                "type": ["CNB", "RP"],
                "label": [0, 1],
            }
        )
        state_a = metadata.copy()
        state_a["prob_1"] = [0.1, 0.8]
        state_b = metadata.copy()
        state_b["prob_1"] = [0.3, 1.0]
        trajectory = trajectory_predictions([state_a, state_b])
        np.testing.assert_allclose(trajectory["prob_1"], [0.2, 0.9])
        np.testing.assert_allclose(trajectory["state_variance"], [0.01, 0.01])

        folds = []
        for offset in [-0.10, -0.05, 0.0, 0.05, 0.10]:
            fold = trajectory.copy()
            fold["prob_1"] = trajectory["prob_1"] + offset
            folds.append(fold)
        architecture = architecture_test_prediction(folds)
        np.testing.assert_allclose(architecture["prob_1"], [0.2, 0.9])
        np.testing.assert_allclose(architecture["state_variance"], [0.01, 0.01])
        np.testing.assert_allclose(architecture["fold_variance"], [0.005, 0.005])


if __name__ == "__main__":
    unittest.main()
