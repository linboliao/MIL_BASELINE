from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ensemble.spe import (
    apply_anchor_confidence_guard,
    apply_anchor_decision_guard,
    architecture_disagreement,
    binary_macro_f1,
    class_patient_equal_sample_weights,
    fit_architecture_weights,
    fit_constrained_linear_stacking,
    patient_equal_sample_weights,
    residual_similarity_matrix,
    select_architectures,
    select_architectures_for_balanced_accuracy,
    select_architectures_for_macro_f1,
    select_development_robust_anchor,
    select_diversity_veto,
    select_sensitivity_constrained_subset,
    select_top1_anchor_with_fallback,
)
from ensemble.ra_spe import fit_ra_spe, predict_ra_spe
from scripts.Diagnosis.spe.run import (
    architecture_test_prediction,
    configured_devices,
    dtfd_positive_probability,
    merge_architecture_test_predictions,
    patient_id_from_slide,
    prepare_inference_model,
    recover_cached_oof_state_variances,
    resolve_cohort_name,
    selected_checkpoints,
    selected_checkpoints_with_details,
    trajectory_predictions,
)
from utils.model_utils import get_model_from_yaml
from utils.yaml_utils import read_yaml
from utils.spe_model_utils import _evenly_spaced_records, _stable_intervals


class TestSPE(unittest.TestCase):
    def test_output_cohort_defaults_and_override(self):
        self.assertEqual(resolve_cohort_name(None, None), "Internal")
        self.assertEqual(
            resolve_cohort_name(Path("external.csv"), None), "External"
        )
        self.assertEqual(
            resolve_cohort_name(Path("external.csv"), "Internal"), "Internal"
        )
        with self.assertRaisesRegex(ValueError, "Internal.*External"):
            resolve_cohort_name(None, "invalid")

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

    def test_constrained_stacking_respects_concentration_limits(self):
        labels = np.tile([0, 1], 20)
        patients = np.array([f"p{index}" for index in range(len(labels))])
        good = np.where(labels == 1, 0.9, 0.1)
        matrix = np.column_stack(
            [good, 0.5 * good + 0.25, 0.45 + 0.1 * good, 1.0 - good]
        )
        fit, _ = fit_constrained_linear_stacking(
            matrix,
            labels,
            patients,
            architecture_names=["a", "b", "c", "d"],
            max_weight=0.4,
            min_effective_members=3.0,
        )
        weights = np.asarray(fit.weights)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)
        self.assertLessEqual(float(weights.max()), 0.4 + 1e-7)
        self.assertGreaterEqual(fit.effective_members, 3.0 - 1e-6)
        self.assertGreater(weights[0], weights[3])

    def test_ra_spe_cross_fits_and_bounds_dynamic_weights(self):
        labels = np.tile([0, 1], 15)
        patients = np.array([f"p{index}" for index in range(len(labels))])
        folds = np.repeat(np.arange(1, 6), 6)
        good = np.where(labels == 1, 0.85, 0.15)
        matrix = np.column_stack(
            [good, np.clip(good + 0.05, 0, 1), 0.55 - 0.1 * labels, 1.0 - good]
        )
        state_variance = np.full_like(matrix, 0.01)
        fit, model, oof_probability, oof_weights = fit_ra_spe(
            matrix,
            labels,
            patients,
            folds,
            architecture_names=["a", "b", "c", "d"],
            state_variances=state_variance,
            hidden_dim=8,
            epochs=15,
            max_weight=0.4,
            seed=7,
        )
        test_probability, test_weights = predict_ra_spe(
            model, matrix, state_variance
        )
        self.assertEqual(oof_probability.shape, (len(labels),))
        self.assertEqual(oof_weights.shape, matrix.shape)
        self.assertTrue(np.isfinite(test_probability).all())
        np.testing.assert_allclose(test_weights.sum(axis=1), 1.0, atol=1e-6)
        self.assertLessEqual(float(test_weights.max()), 0.4 + 1e-6)
        self.assertTrue(fit.state_variance_used)

    def test_recovers_state_variance_from_legacy_refit_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member_root = root / "member_predictions" / "a" / "run_1"
            member_root.mkdir(parents=True)
            base = pd.DataFrame(
                {
                    "slide_id": ["s1", "s2"],
                    "patient_id": ["p1", "p2"],
                    "label": [0, 1],
                    "fold": [1, 2],
                    "prob_a": [0.1, 0.9],
                }
            )
            member = base.rename(columns={"prob_a": "prob_1"})
            member["state_variance"] = [0.01, 0.02]
            member.to_csv(member_root / "oof_predictions.csv", index=False)
            recovered = recover_cached_oof_state_variances(
                base,
                root,
                ["a"],
                {"a": {"training_run": "some/path/run_1"}},
            )
            np.testing.assert_allclose(
                recovered["state_variance_a"], [0.01, 0.02]
            )

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

    def test_binary_macro_f1_and_architecture_selection(self):
        score = binary_macro_f1(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.8, 0.9, 0.7],
            threshold=0.5,
        )
        self.assertAlmostEqual(score, (2 / 3 + 4 / 5) / 2)

        labels = np.tile([0, 1], 20)
        patients = np.array([f"p{index}" for index in range(len(labels))])
        folds = np.repeat(np.arange(1, 6), 8)
        good = np.where(labels == 1, 0.8, 0.2).astype(float)
        good[::10] = 0.8
        complement = np.where(labels == 1, 0.75, 0.25).astype(float)
        complement[::10] = 0.05
        unsuitable = 1.0 - np.where(labels == 1, 0.8, 0.2)
        selection = select_architectures_for_macro_f1(
            np.column_stack([good, complement, unsuitable]),
            labels,
            patients,
            folds,
            ["good", "complement", "unsuitable"],
            min_members=2,
            max_members=3,
            min_cv_improvement=1e-4,
            max_individual_macro_f1_gap=0.11,
        )
        self.assertEqual(selection.selected_names, ("complement", "good"))
        self.assertNotIn("unsuitable", selection.eligible_names)
        self.assertGreater(selection.cross_validated_macro_f1, 0.9)

    def test_diversity_veto_selects_accurate_anchor_and_complement(self):
        labels = np.tile([0, 1], 20)
        patients = np.array([f"p{index}" for index in range(len(labels))])
        anchor = np.where(labels == 1, 0.9, 0.1).astype(float)
        similar = np.clip(anchor + np.tile([0.02, -0.02], 20), 0, 1)
        complement = anchor.copy()
        complement[[0, 10, 21, 31]] = 1.0 - complement[[0, 10, 21, 31]]
        selection = select_diversity_veto(
            np.column_stack([anchor, similar, complement]),
            labels,
            patients,
            ["anchor", "similar", "complement"],
            max_veto_bacc_gap=0.10,
        )
        self.assertEqual(selection.anchor_name, "anchor")
        self.assertEqual(selection.veto_name, "complement")
        self.assertEqual(selection.selected_names, ("anchor", "complement"))

    def test_sensitivity_constrained_subset_respects_reference(self):
        labels = np.tile([0, 1], 20)
        reference = np.where(labels == 1, 0.8, 0.2).astype(float)
        reference[[1, 3]] = 0.4
        noisy = np.where(labels == 1, 0.9, 0.45).astype(float)
        specific = np.where(labels == 1, 0.7, 0.1).astype(float)
        selection = select_sensitivity_constrained_subset(
            np.column_stack([noisy, specific]),
            labels,
            ["noisy", "specific"],
            reference,
            min_members=1,
            max_members=2,
        )
        self.assertGreaterEqual(
            selection.cross_validated_sensitivity,
            selection.reference_sensitivity,
        )
        self.assertGreaterEqual(selection.cross_validated_specificity, 0.95)
        self.assertTrue(0.0 < selection.classification_threshold < 1.0)

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

    @staticmethod
    def _write_checkpoint_trajectory(
        fold_dir: Path,
        rows: list[tuple[float, float, int, int, int, int]],
    ) -> None:
        records = []
        for epoch, (bacc, macro_f1, tn, fp, fn, tp) in enumerate(rows, start=1):
            relative = f"epoch_checkpoints/Epoch_{epoch:04d}.pth"
            checkpoint = fold_dir / relative
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.touch()
            records.append(
                {
                    "epoch": epoch,
                    "checkpoint": relative,
                    "val_metrics": {
                        "bacc": bacc,
                        "macro_f1": macro_f1,
                        "confusion_mat": [[tn, fp], [fn, tp]],
                    },
                }
            )
        (fold_dir / "checkpoint_manifest.json").write_text(
            json.dumps({"epochs": records}), encoding="utf-8"
        )

    def test_high_performance_stable_basin_selects_contiguous_plateau(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory)
            self._write_checkpoint_trajectory(
                fold_dir,
                [
                    (0.970, 0.969, 96, 4, 2, 98),
                    (0.981, 0.980, 97, 3, 1, 99),
                    (0.980, 0.979, 97, 3, 1, 99),
                    (0.979, 0.978, 97, 3, 1, 99),
                    (0.980, 0.979, 97, 3, 1, 99),
                    (0.981, 0.980, 97, 3, 1, 99),
                    (0.975, 0.974, 95, 5, 0, 100),
                ],
            )
            selected, details = selected_checkpoints_with_details(
                fold_dir,
                {
                    "strategy": "high_performance_stable_basin",
                    "metric": "bacc",
                    "metric_tolerance": 0.005,
                    "secondary_metric": "macro_f1",
                    "secondary_metric_tolerance": 0.005,
                    "min_consecutive": 5,
                    "metric_range_tolerance": 0.003,
                    "sensitivity_range_tolerance": 0.01,
                    "specificity_range_tolerance": 0.01,
                    "max_checkpoints": 3,
                },
            )
            self.assertEqual(details["selection_tier"], "strict_stable_basin")
            self.assertFalse(details["fallback_used"])
            self.assertEqual(details["retained_interval"]["start_epoch"], 2)
            self.assertEqual(details["retained_interval"]["end_epoch"], 6)
            self.assertEqual(
                [path.stem for path in selected],
                ["Epoch_0002", "Epoch_0004", "Epoch_0006"],
            )

    def test_stable_basin_relaxes_class_ranges_before_performance_band(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory)
            self._write_checkpoint_trajectory(
                fold_dir,
                [
                    (0.981, 0.980, 99, 1, 3, 97),
                    (0.980, 0.979, 97, 3, 1, 99),
                    (0.979, 0.978, 99, 1, 3, 97),
                    (0.980, 0.979, 97, 3, 1, 99),
                    (0.981, 0.980, 99, 1, 3, 97),
                ],
            )
            _, details = selected_checkpoints_with_details(
                fold_dir,
                {
                    "strategy": "high_performance_stable_basin",
                    "min_consecutive": 5,
                    "metric_range_tolerance": 0.003,
                    "sensitivity_range_tolerance": 0.01,
                    "specificity_range_tolerance": 0.01,
                },
            )
            self.assertEqual(
                details["selection_tier"], "class_relaxed_stable_basin"
            )
            self.assertTrue(details["fallback_used"])

    def test_stable_basin_falls_back_to_high_performance_band(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory)
            self._write_checkpoint_trajectory(
                fold_dir,
                [
                    (0.981, 0.980, 98, 2, 2, 98),
                    (0.970, 0.969, 98, 2, 2, 98),
                    (0.980, 0.979, 98, 2, 2, 98),
                    (0.969, 0.968, 98, 2, 2, 98),
                    (0.979, 0.978, 98, 2, 2, 98),
                ],
            )
            selected, details = selected_checkpoints_with_details(
                fold_dir,
                {
                    "strategy": "high_performance_stable_basin",
                    "min_consecutive": 3,
                    "max_checkpoints": 3,
                },
            )
            self.assertEqual(
                details["selection_tier"], "high_performance_band_fallback"
            )
            self.assertTrue(details["fallback_used"])
            self.assertEqual(
                [path.stem for path in selected],
                ["Epoch_0001", "Epoch_0003", "Epoch_0005"],
            )

    def test_best_state_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            fold_dir = Path(directory)
            checkpoint = fold_dir / "Best_EPOCH_17.pth"
            checkpoint.touch()
            self.assertEqual(
                selected_checkpoints(fold_dir, {"strategy": "best_state"}),
                [checkpoint],
            )

    def test_top1_anchor_deploys_only_after_guardrails_pass(self):
        labels = np.tile([0, 0, 1, 1], 5)
        folds = np.repeat(np.arange(1, 6), 4)
        patients = np.asarray([f"p{index}" for index in range(len(labels))])
        anchor = np.tile([0.49, 0.10, 0.49, 0.90], 5)
        complement = np.tile([0.10, 0.90, 0.90, 0.10], 5)
        weak = np.full(len(labels), 0.5)
        selection, deployed_oof, proposed_oof = select_top1_anchor_with_fallback(
            np.column_stack([anchor, complement, weak]),
            labels,
            patients,
            folds,
            ["anchor", "complement", "weak"],
            min_members=2,
            max_members=2,
            max_individual_bacc_gap=0.30,
            min_oof_bacc_gain=0.005,
            min_non_decreasing_folds=4,
        )
        self.assertEqual(selection.anchor_name, "anchor")
        self.assertFalse(selection.fallback_triggered)
        self.assertEqual(selection.deployed_names, ("anchor", "complement"))
        self.assertGreaterEqual(selection.deployed_weights[0], 0.70)
        np.testing.assert_allclose(deployed_oof, proposed_oof)

    def test_top1_anchor_falls_back_when_there_is_no_gain(self):
        labels = np.tile([0, 1], 5)
        folds = np.repeat(np.arange(1, 6), 2)
        patients = np.asarray([f"p{index}" for index in range(len(labels))])
        anchor = np.tile([0.1, 0.9], 5)
        selection, deployed_oof, _ = select_top1_anchor_with_fallback(
            np.column_stack([anchor, anchor]),
            labels,
            patients,
            folds,
            ["anchor", "copy"],
            min_members=2,
            max_members=2,
            max_individual_bacc_gap=None,
        )
        self.assertTrue(selection.fallback_triggered)
        self.assertEqual(selection.deployed_names, ("anchor",))
        np.testing.assert_allclose(selection.deployed_weights, [1.0, 0.0])
        np.testing.assert_allclose(deployed_oof, anchor)

    def test_fixed_clam_anchor_uses_confidence_and_sensitivity_guards(self):
        labels = np.tile([0, 0, 1, 1], 5)
        folds = np.repeat(np.arange(1, 6), 4)
        patients = np.asarray([f"p{index}" for index in range(len(labels))])
        clam = np.tile([0.1, 0.49, 0.9, 0.49], 5)
        complement = np.tile([0.1, 0.1, 0.9, 0.9], 5)
        selection, deployed_oof, _ = select_top1_anchor_with_fallback(
            np.column_stack([clam, complement]),
            labels,
            patients,
            folds,
            ["CLAM_SB_MIL", "better_oof_model"],
            fixed_anchor_name="CLAM_SB_MIL",
            anchor_min_weight=0.8,
            anchor_confidence_low=0.2,
            anchor_confidence_high=0.8,
            min_members=2,
            max_members=2,
            max_individual_bacc_gap=None,
            max_individual_sensitivity_gap=0.01,
            bacc_noninferiority_margin=0.003,
            max_oof_sensitivity_drop=0.0,
            max_oof_specificity_drop=0.01,
            min_non_decreasing_folds=4,
            min_non_decreasing_sensitivity_folds=4,
        )
        self.assertEqual(selection.anchor_name, "CLAM_SB_MIL")
        self.assertTrue(selection.fixed_anchor)
        self.assertFalse(selection.fallback_triggered)
        self.assertGreaterEqual(selection.deployed_weights[0], 0.8)
        self.assertGreaterEqual(selection.sensitivity_gain, 0.0)
        self.assertEqual(
            selection.selection_objective,
            "bacc_noninferiority_then_macro_f1_roc_auc_average_precision_log_loss",
        )
        self.assertGreaterEqual(
            selection.balanced_accuracy_gain,
            -selection.bacc_noninferiority_margin,
        )
        high_confidence = (clam <= 0.2) | (clam >= 0.8)
        np.testing.assert_allclose(deployed_oof[high_confidence], clam[high_confidence])

        guarded = apply_anchor_confidence_guard(clam, complement, 0.2, 0.8)
        np.testing.assert_allclose(guarded[high_confidence], clam[high_confidence])

        decision_guarded = apply_anchor_decision_guard(
            [0.51, 0.51],
            [0.49, 0.49],
            np.asarray([[0.40, 0.30], [0.40, 0.60]]),
            classification_threshold=0.5,
            confidence_low=0.2,
            confidence_high=0.8,
            min_override_agreement=2,
        )
        np.testing.assert_allclose(decision_guarded, [0.49, 0.51])

    def test_development_robust_anchor_uses_equivalence_and_trajectory(self):
        labels = np.tile([0, 0, 1, 1], 5)
        folds = np.repeat(np.arange(1, 6), 4)
        patients = np.asarray([f"p{index}" for index in range(len(labels))])
        top_bacc = np.tile([0.01, 0.49, 0.51, 0.99], 5)
        calibrated = np.tile([0.10, 0.20, 0.80, 0.90], 5)
        far_worse = 1.0 - calibrated
        probabilities = np.column_stack([top_bacc, calibrated, far_worse])
        state_variances = np.column_stack(
            [
                np.full(len(labels), 0.020),
                np.full(len(labels), 0.001),
                np.full(len(labels), 0.0001),
            ]
        )
        names = ["A_top_bacc", "B_calibrated", "C_far_worse"]

        selection = select_development_robust_anchor(
            probabilities,
            labels,
            patients,
            folds,
            names,
            state_variances=state_variances,
            bacc_noninferiority_margin=0.002,
            max_sensitivity_drop=0.005,
        )
        self.assertEqual(selection.best_bacc_name, "A_top_bacc")
        self.assertEqual(selection.selected_name, "B_calibrated")
        self.assertEqual(
            selection.eligible_names, ("A_top_bacc", "B_calibrated")
        )
        self.assertTrue(selection.state_variance_used)
        self.assertLess(
            selection.class_patient_equal_log_losses[1],
            selection.class_patient_equal_log_losses[0],
        )
        self.assertNotIn("C_far_worse", selection.accuracy_eligible_names)

        guard, deployed_oof, _ = select_top1_anchor_with_fallback(
            probabilities,
            labels,
            patients,
            folds,
            names,
            anchor_selection_strategy="bacc_equivalent_calibrated_stable",
            state_variances=state_variances,
            min_members=1,
            max_members=1,
            min_oof_bacc_gain=0.0,
            min_non_decreasing_folds=5,
        )
        self.assertEqual(guard.anchor_name, "B_calibrated")
        self.assertFalse(guard.fixed_anchor)
        self.assertEqual(
            guard.anchor_selection_strategy,
            "bacc_equivalent_calibrated_stable",
        )
        self.assertIsNotNone(guard.anchor_selection_diagnostics)
        np.testing.assert_allclose(deployed_oof, calibrated)

        invalid_variances = state_variances.copy()
        invalid_variances[0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            select_development_robust_anchor(
                probabilities,
                labels,
                patients,
                folds,
                names,
                state_variances=invalid_variances,
            )

    def test_development_maximin_anchor_prioritizes_worst_fold(self):
        labels = np.tile([0, 0, 1, 1], 5)
        folds = np.repeat(np.arange(1, 6), 4)
        patients = np.asarray([f"p{index}" for index in range(len(labels))])
        unstable = np.tile([0.10, 0.20, 0.80, 0.90], 5)
        unstable[:4] = [0.90, 0.20, 0.80, 0.90]
        stable = np.tile([0.15, 0.25, 0.75, 0.85], 5)
        probabilities = np.column_stack([unstable, stable])
        variances = np.column_stack(
            [np.full(len(labels), 0.0001), np.full(len(labels), 0.01)]
        )
        calibrated = select_development_robust_anchor(
            probabilities,
            labels,
            patients,
            folds,
            ["unstable", "stable"],
            state_variances=variances,
            bacc_noninferiority_margin=0.06,
            max_sensitivity_drop=None,
        )
        maximin = select_development_robust_anchor(
            probabilities,
            labels,
            patients,
            folds,
            ["unstable", "stable"],
            state_variances=variances,
            bacc_noninferiority_margin=0.06,
            max_sensitivity_drop=None,
            selection_strategy="bacc_equivalent_maximin_stable",
        )
        self.assertEqual(maximin.selected_name, "stable")
        self.assertGreaterEqual(
            maximin.worst_fold_balanced_accuracies[1],
            maximin.worst_fold_balanced_accuracies[0],
        )
        self.assertEqual(maximin.selection_order[2], "worst_fold_bacc")
        self.assertEqual(
            maximin.as_dict()["strategy"],
            "bacc_equivalent_maximin_stable",
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

    def test_external_patient_id_conventions(self):
        self.assertEqual(patient_id_from_slide("B1858553-7"), "B1858553")
        self.assertEqual(patient_id_from_slide("B1866747-8(1)"), "B1866747")
        self.assertEqual(patient_id_from_slide("202330145.37"), "202330145")
        self.assertEqual(patient_id_from_slide("2025-16710-13.14"), "2025-16710")
        self.assertEqual(patient_id_from_slide("X2025-22069-5"), "X2025-22069")


if __name__ == "__main__":
    unittest.main()
