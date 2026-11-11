from typing import Dict, Tuple
import numpy as np
import pytest

import pde.calibration
from pde.calibration import ConformalCalibrator, SelectiveClassifier


@pytest.mark.unit
class TestCalibrationModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.calibration.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains ConformalCalibrator and SelectiveClassifier.
        """
        expected = ["ConformalCalibrator", "SelectiveClassifier"]
        assert sorted(pde.calibration.__all__) == sorted(expected)
        assert pde.calibration.ConformalCalibrator is ConformalCalibrator
        assert pde.calibration.SelectiveClassifier is SelectiveClassifier


@pytest.mark.unit
class TestConformalCalibrator:
    """
    Unit test suite for ConformalCalibrator fitting, p-value computation, and Bonferroni fusion.
    """

    def test_init_state(self) -> None:
        """
        Verifies initial state of an uncalibrated calibrator.
        """
        calibrator = ConformalCalibrator()
        assert isinstance(calibrator.reference_scores_, np.ndarray)
        assert len(calibrator.reference_scores_) == 0
        assert not calibrator.is_calibrated_

    def test_fit_with_valid_scores(self) -> None:
        """
        Verifies that fitting sorts and stores reference scores and sets is_calibrated_ to True.
        """
        calibrator = ConformalCalibrator()
        scores = [5.0, 1.2, 3.4, 0.8]
        result = calibrator.fit(scores)

        assert result is calibrator
        assert calibrator.is_calibrated_
        assert np.array_equal(calibrator.reference_scores_, np.array([0.8, 1.2, 3.4, 5.0]))

    def test_fit_raises_value_error_on_empty_scores(self) -> None:
        """
        Ensures fitting with empty reference scores raises a ValueError.
        """
        calibrator = ConformalCalibrator()
        with pytest.raises(ValueError, match="cannot be empty"):
            calibrator.fit([])

    def test_compute_pvalue_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures compute_pvalue raises a RuntimeError if called prior to fit.
        """
        calibrator = ConformalCalibrator()
        with pytest.raises(RuntimeError, match="must be adjusted using .fit\\(\\)"):
            calibrator.compute_pvalue([1.0, 2.0])

    def test_compute_pvalue_mathematical_correctness(self) -> None:
        """
        Verifies conformal p-value calculation against exact mathematical definition:
            p(s) = (1 + sum(s_cal >= s)) / (N_cal + 1)
        """
        ref_scores = [1.0, 2.0, 3.0, 4.0]
        n_ref = len(ref_scores)
        calibrator = ConformalCalibrator().fit(ref_scores)

        p_low = calibrator.compute_pvalue(0.5)
        assert np.isclose(p_low[0], 1.0)

        p_high = calibrator.compute_pvalue(5.0)
        assert np.isclose(p_high[0], 1.0 / (n_ref + 1))

        p_exact = calibrator.compute_pvalue(2.0)
        assert np.isclose(p_exact[0], 0.8)

        p_mid = calibrator.compute_pvalue(2.5)
        assert np.isclose(p_mid[0], 0.6)

    def test_compute_pvalue_accepts_scalar_and_sequence(self) -> None:
        """
        Verifies compute_pvalue handles scalar float, list, and numpy array inputs.
        """
        calibrator = ConformalCalibrator().fit([10.0, 20.0, 30.0])

        res_scalar = calibrator.compute_pvalue(15.0)
        assert isinstance(res_scalar, np.ndarray)
        assert res_scalar.shape == (1,)

        res_list = calibrator.compute_pvalue([15.0, 25.0])
        assert isinstance(res_list, np.ndarray)
        assert res_list.shape == (2,)

        res_arr = calibrator.compute_pvalue(np.array([15.0, 25.0]))
        assert isinstance(res_arr, np.ndarray)
        assert res_arr.shape == (2,)

    def test_bonferroni_combine_1d_input(self) -> None:
        """
        Verifies that passing a 1D p-values array returns the array unchanged.
        """
        p_vals = np.array([0.1, 0.4, 0.02])
        combined = ConformalCalibrator.bonferroni_combine(p_vals)
        assert np.array_equal(combined, p_vals)

    def test_bonferroni_combine_multi_detector_matrix(self) -> None:
        """
        Verifies row-wise Bonferroni correction: p_ens = min(1.0, M * min(p_1, ..., p_M)).
        """
        p_matrix = np.array(
            [
                [0.01, 0.05],  # M=2, min=0.01 -> 0.02
                [0.40, 0.60],  # M=2, min=0.40 -> 0.80
                [0.70, 0.80],  # M=2, min=0.70 -> 1.40 bounded to 1.0
                [0.00, 0.10],  # M=2, min=0.00 -> 0.00
            ]
        )
        combined = ConformalCalibrator.bonferroni_combine(p_matrix)

        expected = np.array([0.02, 0.80, 1.00, 0.00])
        assert np.allclose(combined, expected)

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """
        Verifies serialization and reconstruction of ConformalCalibrator.
        """
        scores = [1.5, 2.5, 3.5, 4.5]
        calibrator = ConformalCalibrator().fit(scores)

        state = calibrator.to_dict()
        assert state["is_calibrated"] is True
        assert state["reference_scores"] == [1.5, 2.5, 3.5, 4.5]

        reconstructed = ConformalCalibrator.from_dict(state)
        assert reconstructed.is_calibrated_ is True
        assert np.array_equal(reconstructed.reference_scores_, calibrator.reference_scores_)

        test_query = [2.0, 4.0]
        assert np.allclose(
            calibrator.compute_pvalue(test_query),
            reconstructed.compute_pvalue(test_query),
        )

    def test_conformal_with_conftest_fixture(
        self,
        conformal_score_distributions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """
        Tests end-to-end conformal calibration and p-value computation using the shared conftest fixture.
        """
        cal_scores, test_scores, y_test = conformal_score_distributions
        calibrator = ConformalCalibrator().fit(cal_scores)
        p_values = calibrator.compute_pvalue(test_scores)

        assert len(p_values) == len(test_scores)
        assert np.all((p_values >= 0.0) & (p_values <= 1.0))

        normal_pvals = p_values[y_test == 0]
        anom_pvals = p_values[y_test == 1]
        assert normal_pvals.mean() > anom_pvals.mean()


@pytest.mark.unit
class TestSelectiveClassifier:
    """
    Unit test suite for SelectiveClassifier thresholding, triage, and performance metrics.
    """

    def test_init_valid_thresholds(self) -> None:
        """
        Verifies valid initialization of alpha and review_boundary.
        """
        clf = SelectiveClassifier(alpha=0.05, review_boundary=0.20)
        assert clf.alpha == 0.05
        assert clf.review_boundary == 0.20

    @pytest.mark.parametrize(
        "alpha,review_boundary",
        [
            (0.0, 0.20),    # alpha <= 0
            (-0.05, 0.20),  # alpha < 0
            (0.25, 0.20),   # alpha >= review_boundary
            (0.20, 0.20),   # alpha == review_boundary
            (0.05, 1.0),    # review_boundary >= 1.0
            (0.05, 1.5),    # review_boundary > 1.0
            (0.05, -0.1),   # review_boundary < 0
        ],
    )
    def test_init_raises_value_error_on_invalid_bounds(
        self, alpha: float, review_boundary: float
    ) -> None:
        """
        Verifies ValueError when the constraint 0 < alpha < review_boundary < 1 is violated.
        """
        with pytest.raises(ValueError, match="0 < alpha"):
            SelectiveClassifier(alpha=alpha, review_boundary=review_boundary)

    def test_decide_classification_rules(self) -> None:
        """
        Verifies strict mapping from p-values to decision strings:
            - p <= alpha -> 'anomaly'
            - alpha < p < review_boundary -> 'review'
            - p >= review_boundary -> 'no_alert'
        """
        clf = SelectiveClassifier(alpha=0.05, review_boundary=0.20)
        p_vals = np.array([0.01, 0.05, 0.05001, 0.10, 0.1999, 0.20, 0.50])
        decisions = clf.decide(p_vals)

        expected = np.array(
            [
                "anomaly",   # 0.01 <= 0.05
                "anomaly",   # 0.05 <= 0.05 (exact boundary)
                "review",    # 0.05001 in (0.05, 0.20)
                "review",    # 0.10 in (0.05, 0.20)
                "review",    # 0.1999 in (0.05, 0.20)
                "no_alert",  # 0.20 >= 0.20 (exact boundary)
                "no_alert",  # 0.50 >= 0.20
            ]
        )
        assert np.array_equal(decisions, expected)

    def test_decide_scalar_input(self) -> None:
        """
        Verifies that decide handles single scalar float inputs.
        """
        clf = SelectiveClassifier(alpha=0.05, review_boundary=0.20)

        assert clf.decide(0.02) == "anomaly"
        assert clf.decide(0.15) == "review"
        assert clf.decide(0.80) == "no_alert"

    def test_evaluate_metrics_standard_case(self) -> None:
        """
        Verifies metric calculation with a balanced mix of decisions.
        """
        clf = SelectiveClassifier(alpha=0.05, review_boundary=0.20)

        ground_truth = [0, 0, 0, 1, 1, 1]
        decisions = [
            "no_alert",  # TN (0, no_alert)
            "anomaly",   # FP (0, anomaly)
            "review",    # Normal in review
            "anomaly",   # TP (1, anomaly)
            "review",    # Reviewed anomaly
            "no_alert",  # FN (1, no_alert)
        ]

        metrics = clf.evaluate_metrics(ground_truth, decisions)

        assert np.isclose(metrics["coverage"], 4.0 / 6.0)
        assert np.isclose(metrics["review_rate"], 2.0 / 6.0)

        assert np.isclose(metrics["precision_automatic"], 0.5)

        assert np.isclose(metrics["recall_automatic"], 1.0 / 3.0)

        assert np.isclose(metrics["operational_detection_rate"], 2.0 / 3.0)

        assert np.isclose(metrics["population_fpr"], 1.0 / 3.0)

    def test_evaluate_metrics_zero_anomalies_edge_case(self) -> None:
        """
        Verifies that when ground truth contains zero anomalies,
        recall and precision metrics gracefully return 0.0 without division by zero errors.
        """
        clf = SelectiveClassifier(alpha=0.05, review_boundary=0.20)
        ground_truth = [0, 0, 0]
        decisions = ["no_alert", "no_alert", "no_alert"]

        metrics = clf.evaluate_metrics(ground_truth, decisions)

        assert metrics["coverage"] == 1.0
        assert metrics["review_rate"] == 0.0
        assert metrics["precision_automatic"] == 0.0
        assert metrics["recall_automatic"] == 0.0
        assert metrics["operational_detection_rate"] == 0.0
        assert metrics["population_fpr"] == 0.0

    def test_evaluate_metrics_zero_normals_edge_case(self) -> None:
        """
        Verifies that when ground truth contains zero normals,
        population_fpr gracefully returns 0.0 without division by zero errors.
        """
        clf = SelectiveClassifier(alpha=0.05, review_boundary=0.20)
        ground_truth = [1, 1]
        decisions = ["anomaly", "anomaly"]

        metrics = clf.evaluate_metrics(ground_truth, decisions)

        assert metrics["population_fpr"] == 0.0
        assert metrics["precision_automatic"] == 1.0
        assert metrics["recall_automatic"] == 1.0
        assert metrics["operational_detection_rate"] == 1.0

    def test_evaluate_metrics_all_reviewed_zero_coverage(self) -> None:
        """
        Verifies metrics when all samples are routed to human review (100% review, 0% coverage).
        """
        clf = SelectiveClassifier(alpha=0.05, review_boundary=0.20)
        ground_truth = [0, 1]
        decisions = ["review", "review"]

        metrics = clf.evaluate_metrics(ground_truth, decisions)

        assert metrics["coverage"] == 0.0
        assert metrics["review_rate"] == 1.0
        assert metrics["precision_automatic"] == 0.0
        assert metrics["recall_automatic"] == 0.0
        assert metrics["operational_detection_rate"] == 1.0
        assert metrics["population_fpr"] == 0.0
