from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ensemble.spe import (
    architecture_disagreement,
    class_patient_equal_sample_weights,
    fit_architecture_weights,
    patient_equal_sample_weights,
    residual_similarity_matrix,
    select_architectures,
    select_architectures_for_balanced_accuracy,
)
from scripts.Diagnosis.spe.run import (
    architecture_test_prediction,
    configured_devices,
    dtfd_positive_probability,
    merge_architecture_test_predictions,
    prepare_inference_model,
    selected_checkpoints,
    trajectory_predictions,
)
from utils.model_utils import get_model_from_yaml
from utils.yaml_utils import read_yaml
from utils.spe_model_utils import _evenly_spaced_records, _stable_intervals


class TestSPE(unittest.TestCase):
    def test_device_configuration(self):
        self.assertEqual(configured_devices({"device": "cpu"}, None), ["cpu"])
        self.assertEqual(
            configured_devices({"devices": ["cpu"]}, "cpu"),
            ["cpu"],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            configured_devices({"devices": ["cpu", "cpu"]}, None)

    def test_dtfd_component_checkpoint_and_forward(self):
        config = read_yaml("configs/Diagnosis/MIL/DTFD_MIL.yaml")
        component_names = ("classifier", "attention", "dimReduction", "attCls")
        components = get_model_from_yaml(config)
        checkpoint = {
            name: component.state_dict()
            for name, component in zip(component_names, components)
        }
        loaded = prepare_inference_model(config, checkpoint, torch.device("cpu"))
        probability = dtfd_positive_probability(
            torch.randn(1, 10, int(config.Model.in_dim)), loaded, config
        )
        self.assertEqual(len(loaded), 4)
        self.assertTrue(0.0 <= float(probability) <= 1.0)

    def test_patient_equal_weights(self):
        patients = np.array(["a", "a", "a", "b"])
        weights = patient_equal_sample_weights(patients)
        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3]), 0.5)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_class_patient_equal_weights(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        patients = np.array(["a", "a", "b", "c", "c", "d"])
        weights = class_patient_equal_sample_weights(labels, patients)
        self.assertAlmostEqual(float(weights[labels == 0].sum()), 0.5)
        self.assertAlmostEqual(float(weights[labels == 1].sum()), 0.5)
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertAlmostEqual(float(weights[3:5].sum()), float(weights[5]))

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

    def test_architecture_selection_filters_unsuitable_members(self):
        labels = np.tile([0, 1], 10)
        patients = np.array([f"p{index}" for index in range(len(labels))])
        folds = np.repeat(np.arange(1, 6), 4)
        good = np.where(labels == 1, 0.85, 0.15)
        complement = np.where(labels == 1, 0.80, 0.20)
        unsuitable = 1.0 - good
        selection = select_architectures(
            np.column_stack([good, complement, unsuitable]),
            labels,
            patients,
            folds,
            ["good", "complement", "unsuitable"],
            min_members=2,
            max_members=3,
            min_cv_improvement=1e-4,
        )
        self.assertEqual(selection.selected_names, ("good", "complement"))
        self.assertNotIn("unsuitable", selection.eligible_names)

    def test_balanced_accuracy_architecture_selection(self):
        labels = np.tile([0, 1], 20)
        patients = np.array([f"p{index}" for index in range(len(labels))])
        folds = np.repeat(np.arange(1, 6), 8)
        good = np.where(labels == 1, 0.8, 0.2).astype(float)
        good[::10] = 0.8
        complement = np.where(labels == 1, 0.75, 0.25).astype(float)
        complement[::10] = 0.05
        unsuitable = 1.0 - np.where(labels == 1, 0.8, 0.2)
        selection = select_architectures_for_balanced_accuracy(
            np.column_stack([good, complement, unsuitable]),
            labels,
            patients,
            folds,
            ["good", "complement", "unsuitable"],
            min_members=2,
            max_members=3,
            min_cv_improvement=1e-4,
        )
        self.assertEqual(selection.selected_names, ("complement", "good"))
        self.assertNotIn("unsuitable", selection.eligible_names)

    def test_high_performance_checkpoint_band(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory)
            values = [0.70, 0.80, 0.798, 0.796, 0.79, 0.799, 0.794]
            records = []
            for epoch, value in enumerate(values, start=1):
                relative = f"epoch_checkpoints/Epoch_{epoch:04d}.pth"
                checkpoint = fold_dir / relative
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                records.append(
                    {
                        "epoch": epoch,
                        "checkpoint": relative,
                        "val_metrics": {"bacc": value},
                    }
                )
            (fold_dir / "checkpoint_manifest.json").write_text(
                json.dumps({"epochs": records}), encoding="utf-8"
            )
            selected = selected_checkpoints(
                fold_dir,
                {
                    "strategy": "high_performance_band",
                    "metric": "bacc",
                    "metric_tolerance": 0.005,
                    "max_checkpoints": 3,
                },
            )
            self.assertEqual(
                [path.stem for path in selected],
                ["Epoch_0002", "Epoch_0004", "Epoch_0006"],
            )

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

    def test_test_architecture_merge_uses_immutable_metadata(self):
        metadata = pd.DataFrame(
            {
                "slide_id": ["s1", "s2"],
                "patient_id": ["p1", "p2"],
                "type": ["CNB", "RP"],
                "label": [0, 1],
            }
        )
        predictions = {}
        for name, values in (("a", [0.1, 0.8]), ("b", [0.2, 0.9])):
            frame = metadata.copy()
            frame["prob_1"] = values
            frame["state_variance"] = [0.01, 0.02]
            frame["fold_variance"] = [0.03, 0.04]
            predictions[name] = frame
        merged = merge_architecture_test_predictions(predictions, ["a", "b"])
        self.assertEqual(merged.shape, (2, 10))
        np.testing.assert_allclose(merged["prob_b"], [0.2, 0.9])


if __name__ == "__main__":
    unittest.main()
