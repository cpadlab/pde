
from typing import Dict, Tuple
import numpy as np
import pytest

import pde.geometry
from pde.geometry import (
    CollectiveCompositionDetector,
    GeometricBonferroniEnsemble,
    GeometricModelsSuite,
)


@pytest.mark.unit
class TestGeometryModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.geometry.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains GeometricModelsSuite, GeometricBonferroniEnsemble,
        and CollectiveCompositionDetector.
        """
        expected = [
            "GeometricModelsSuite",
            "GeometricBonferroniEnsemble",
            "CollectiveCompositionDetector",
        ]
        assert sorted(pde.geometry.__all__) == sorted(expected)
        assert pde.geometry.GeometricModelsSuite is GeometricModelsSuite
        assert pde.geometry.GeometricBonferroniEnsemble is GeometricBonferroniEnsemble
        assert pde.geometry.CollectiveCompositionDetector is CollectiveCompositionDetector


@pytest.mark.unit
class TestGeometricModelsSuite:
    """
    Unit test suite for GeometricModelsSuite fitting, raw score computation, and serialization.
    """

    def test_init_state(self) -> None:
        """
        Verifies default parameter initialization.
        """
        suite = GeometricModelsSuite(random_state=123)
        assert suite.random_state == 123
        assert suite.scaler_ is None
        assert suite.models_ == {}
        assert not suite.is_fitted_

    def test_fit_raises_value_error_on_non_2d_array(self) -> None:
        """
        Ensures fit raises ValueError when X_train is 1D or 3D.
        """
        suite = GeometricModelsSuite()
        with pytest.raises(ValueError, match="must be two-dimensional"):
            suite.fit(np.array([1.0, 2.0, 3.0]))

        with pytest.raises(ValueError, match="must be two-dimensional"):
            suite.fit(np.zeros((5, 3, 2)))

    def test_fit_and_compute_raw_scores_success(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies successful fitting of the four models and score computation.
        """
        X_train = rng.normal(0.0, 1.0, size=(60, 3))
        X_test = rng.normal(0.0, 1.0, size=(10, 3))

        suite = GeometricModelsSuite(random_state=42)
        suite.fit(X_train)

        assert suite.is_fitted_
        assert suite.scaler_ is not None
        assert set(suite.models_.keys()) == {"HDBSCAN", "Isolation Forest", "LOF", "One-Class SVM"}

        scores = suite.compute_raw_scores(X_test)
        assert set(scores.keys()) == {"HDBSCAN", "Isolation Forest", "LOF", "One-Class SVM"}

        for name, arr in scores.items():
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (10,)
            assert not np.isnan(arr).any()

    def test_compute_raw_scores_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures compute_raw_scores raises RuntimeError if called prior to fit.
        """
        suite = GeometricModelsSuite()
        with pytest.raises(RuntimeError, match="must execute .fit\\(\\)"):
            suite.compute_raw_scores(np.zeros((5, 2)))

    def test_to_dict_and_from_dict_roundtrip(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies complete joblib/base64 serialization and reconstruction fidelity.
        """
        X_train = rng.normal(0.0, 1.0, size=(50, 3))
        X_query = rng.normal(0.0, 1.0, size=(5, 3))

        suite = GeometricModelsSuite(random_state=42).fit(X_train)
        state_dict = suite.to_dict()

        assert state_dict["random_state"] == 42
        assert state_dict["is_fitted"] is True
        assert isinstance(state_dict["payload"], str)
        assert len(state_dict["payload"]) > 0

        reconstructed = GeometricModelsSuite.from_dict(state_dict)
        assert reconstructed.is_fitted_ is True
        assert reconstructed.random_state == 42
        assert reconstructed.scaler_ is not None
        assert set(reconstructed.models_.keys()) == set(suite.models_.keys())

        orig_scores = suite.compute_raw_scores(X_query)
        recon_scores = reconstructed.compute_raw_scores(X_query)

        for algo in ["HDBSCAN", "Isolation Forest", "LOF", "One-Class SVM"]:
            assert np.allclose(orig_scores[algo], recon_scores[algo])

    def test_from_dict_empty_payload_fallback(self) -> None:
        """
        Verifies from_dict handles empty or missing payload dictionaries safely.
        """
        empty_data = {"random_state": 99, "is_fitted": False, "payload": ""}
        instance = GeometricModelsSuite.from_dict(empty_data)
        assert instance.random_state == 99
        assert not instance.is_fitted_
        assert instance.scaler_ is None
        assert instance.models_ == {}


@pytest.mark.unit
class TestGeometricBonferroniEnsemble:
    """
    Unit test suite for GeometricBonferroniEnsemble calibration and aggregated inference.
    """

    def test_init_state(self) -> None:
        """
        Verifies default initialization of algorithms and calibrator mapping.
        """
        ensemble = GeometricBonferroniEnsemble(random_state=42)
        assert len(ensemble.ALGORITHMS) == 4
        assert not ensemble.is_calibrated_
        assert set(ensemble.calibrators_.keys()) == set(ensemble.ALGORITHMS)

    def test_predict_pvalues_raises_runtime_error_if_not_calibrated(self) -> None:
        """
        Ensures predict_pvalues raises RuntimeError if called before fit.
        """
        ensemble = GeometricBonferroniEnsemble()
        with pytest.raises(RuntimeError, match="must be calibrated using .fit\\(\\)"):
            ensemble.predict_pvalues(np.zeros((5, 2)))

    def test_fit_and_predict_pvalues_with_fixture(
        self,
        continuous_numpy_partitions: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """
        Verifies full ensemble fitting, calibration, and Bonferroni aggregation using conftest fixture.
        """
        X_train, X_cal, X_test_norm, X_anom = continuous_numpy_partitions
        X_query = np.vstack([X_test_norm[:10], X_anom[:5]])

        ensemble = GeometricBonferroniEnsemble(random_state=42)
        fitted = ensemble.fit(X_train, X_cal)

        assert fitted is ensemble
        assert ensemble.is_calibrated_

        for algo in ensemble.ALGORITHMS:
            assert ensemble.calibrators_[algo].is_calibrated_

        pvalues = ensemble.predict_pvalues(X_query)

        expected_keys = {
            "HDBSCAN",
            "Isolation Forest",
            "LOF",
            "One-Class SVM",
            "Bonferroni Ensemble",
        }
        assert set(pvalues.keys()) == expected_keys

        for key, arr in pvalues.items():
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (15,)
            assert np.all((arr >= 0.0) & (arr <= 1.0))

        p_mat = np.column_stack([pvalues[algo] for algo in ensemble.ALGORITHMS])
        min_p = np.min(p_mat, axis=1)
        expected_bonf = np.minimum(1.0, 4 * min_p)
        assert np.allclose(pvalues["Bonferroni Ensemble"], expected_bonf)


@pytest.mark.unit
class TestCollectiveCompositionDetector:
    """
    Unit test suite for CollectiveCompositionDetector mixture fitting and divergence scoring.
    """

    def test_init_parameters(self) -> None:
        """
        Verifies default parameter values.
        """
        detector = CollectiveCompositionDetector(n_clusters=4, random_state=101)
        assert detector.n_clusters == 4
        assert detector.random_state == 101
        assert detector.kmeans_ is None
        assert detector.baseline_proportions_ is None
        assert not detector.is_fitted_

    def test_evaluate_group_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures evaluate_group raises RuntimeError prior to fitting.
        """
        detector = CollectiveCompositionDetector()
        with pytest.raises(RuntimeError, match="call .fit\\(\\) before evaluating"):
            detector.evaluate_group(np.zeros((5, 2)))

    def test_fit_and_evaluate_group_metrics(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies TV distance, JS divergence, Chi-Square statistic, and observed proportions.
        """
        c1 = rng.normal(-5.0, 0.5, size=(40, 2))
        c2 = rng.normal(0.0, 0.5, size=(40, 2))
        c3 = rng.normal(5.0, 0.5, size=(40, 2))
        X_train = np.vstack([c1, c2, c3])

        detector = CollectiveCompositionDetector(n_clusters=3, random_state=42)
        detector.fit(X_train)

        assert detector.is_fitted_
        assert detector.kmeans_ is not None
        assert detector.baseline_proportions_ is not None
        assert len(detector.baseline_proportions_) == 3
        assert np.isclose(detector.baseline_proportions_.sum(), 1.0)

        group_nominal = np.vstack([
            rng.normal(-5.0, 0.5, size=(10, 2)),
            rng.normal(0.0, 0.5, size=(10, 2)),
            rng.normal(5.0, 0.5, size=(10, 2)),
        ])
        res_nominal = detector.evaluate_group(group_nominal)

        assert set(res_nominal.keys()) == {
            "total_variation",
            "jensen_shannon_divergence",
            "chi_square_statistic",
            "observed_proportions",
        }
        assert 0.0 <= res_nominal["total_variation"] <= 1.0
        assert 0.0 <= res_nominal["jensen_shannon_divergence"] <= np.log(2)
        assert res_nominal["chi_square_statistic"] >= 0.0
        assert len(res_nominal["observed_proportions"]) == 3

        group_skewed = rng.normal(-5.0, 0.5, size=(30, 2))
        res_skewed = detector.evaluate_group(group_skewed)

        assert res_skewed["total_variation"] > res_nominal["total_variation"]
        assert res_skewed["jensen_shannon_divergence"] > res_nominal["jensen_shannon_divergence"]
        assert res_skewed["chi_square_statistic"] > res_nominal["chi_square_statistic"]
