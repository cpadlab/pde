
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest

from pde import PatternDiscoveryEngine


@pytest.mark.integration
class TestEngineApiLifecycle:
    """
    Integration test suite executing fit() -> predict() cycles and verifying output contract consistency.
    """

    def test_unfitted_engine_raises_runtime_error(self) -> None:
        """
        Ensures predict and save raise RuntimeError if invoked before fit().
        """
        engine = PatternDiscoveryEngine()
        assert not engine.is_fitted

        with pytest.raises(RuntimeError, match="must execute .fit\\(\\)"):
            engine.predict(pd.DataFrame({"x": [1.0, 2.0]}))

        with pytest.raises(RuntimeError, match="cannot save an untrained model"):
            engine.save("dummy_path.json")

    def test_continuous_tabular_lifecycle(
        self,
        continuous_tabular_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """
        Executes full fit -> predict lifecycle on continuous tabular data with independent calibration.
        """
        df_train, df_cal, df_test = continuous_tabular_splits
        engine = PatternDiscoveryEngine(alpha=0.05, review_boundary=0.20, random_state=42)

        fitted_engine = engine.fit(df_train, cal_data=df_cal)
        assert fitted_engine is engine
        assert engine.is_fitted is True
        assert engine.modality == "tabular_continuous"
        assert engine.pipeline_name == "Continuous_Conformal_Geometric_Engine"
        assert engine.continuous_suite is not None
        assert engine.continuous_suite.is_fitted_ is True
        assert engine.calibrator.is_calibrated_ is True
        assert len(engine.feature_names) == 3

        # Execute prediction on test split
        preds = engine.predict(df_test)

        assert isinstance(preds, dict)
        assert set(preds.keys()) == {"scores", "p_values", "decisions", "is_anomaly"}

        n_samples = len(df_test)
        for key in ["scores", "p_values", "decisions", "is_anomaly"]:
            assert isinstance(preds[key], np.ndarray)
            assert len(preds[key]) == n_samples

        assert np.all((preds["p_values"] >= 0.0) & (preds["p_values"] <= 1.0))

        valid_labels = {"anomaly", "review", "no_alert"}
        assert set(preds["decisions"]).issubset(valid_labels)

        expected_binary = (preds["decisions"] == "anomaly").astype(int)
        assert np.array_equal(preds["is_anomaly"], expected_binary)

        for p_val, dec in zip(preds["p_values"], preds["decisions"]):
            if p_val <= engine.alpha:
                assert dec == "anomaly"
            elif p_val >= engine.review_boundary:
                assert dec == "no_alert"
            else:
                assert dec == "review"

    def test_heterogeneous_tabular_lifecycle(
        self,
        heterogeneous_tabular_data: Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]],
    ) -> None:
        """
        Executes full fit -> predict lifecycle on mixed-type tabular data.
        """
        train_df, test_df, feature_types = heterogeneous_tabular_data
        engine = PatternDiscoveryEngine(alpha=0.05, review_boundary=0.20, random_state=42)

        engine.fit(train_df)
        assert engine.is_fitted is True
        assert engine.modality == "tabular_heterogeneous"
        assert engine.pipeline_name == "Heterogeneous_Gower_KMedoids_Engine"
        assert engine.gower_metric is not None
        assert engine.gower_metric.is_fitted_ is True
        assert engine.train_heterogeneous_ref_ is not None
        assert engine.calibrator.is_calibrated_ is True

        preds = engine.predict(test_df)

        assert isinstance(preds, dict)
        assert len(preds["scores"]) == len(test_df)
        assert np.all((preds["p_values"] >= 0.0) & (preds["p_values"] <= 1.0))
        assert set(preds["decisions"]).issubset({"anomaly", "review", "no_alert"})
        assert np.array_equal(preds["is_anomaly"], (preds["decisions"] == "anomaly").astype(int))

    def test_sequential_temporal_lifecycle(
        self,
        sequential_temporal_data: Tuple[
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float], str]],
        ],
    ) -> None:
        """
        Executes full fit -> predict lifecycle on sequential temporal trajectories.
        """
        train_seqs, test_normal, anomalies = sequential_temporal_data
        test_queries = test_normal + [(a[0], a[1]) for a in anomalies]

        engine = PatternDiscoveryEngine(alpha=0.05, review_boundary=0.20, random_state=42)

        engine.fit(train_seqs)
        assert engine.is_fitted is True
        assert engine.modality == "sequential_temporal"
        assert engine.pipeline_name == "Semi_Markov_Conformal_Engine"
        assert engine.semi_markov_suite is not None
        assert engine.semi_markov_suite.is_fitted_ is True
        assert engine.calibrator.is_calibrated_ is True

        preds = engine.predict(test_queries)

        assert isinstance(preds, dict)
        assert len(preds["scores"]) == len(test_queries)
        assert np.all((preds["p_values"] >= 0.0) & (preds["p_values"] <= 1.0))
        assert set(preds["decisions"]).issubset({"anomaly", "review", "no_alert"})
        assert np.array_equal(preds["is_anomaly"], (preds["decisions"] == "anomaly").astype(int))

    def test_fit_with_omitted_calibration_data_fallback(
        self,
        continuous_tabular_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """
        Verifies that omitting cal_data (cal_data=None) defaults to using train_data for calibration.
        """
        df_train, _, df_test = continuous_tabular_splits
        engine = PatternDiscoveryEngine(alpha=0.05, review_boundary=0.20)

        engine.fit(df_train, cal_data=None)
        assert engine.is_fitted is True
        assert engine.calibrator.is_calibrated_ is True

        preds = engine.predict(df_test)
        assert len(preds["p_values"]) == len(df_test)

    def test_monitor_drift_stationary_window(
        self,
        continuous_tabular_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """
        Verifies drift monitoring over a stationary in-distribution window.
        """
        from sklearn.datasets import make_blobs

        df_train, df_cal, _ = continuous_tabular_splits
        engine = PatternDiscoveryEngine(alpha=0.05, auto_adapt=True, random_state=42)
        engine.fit(df_train, cal_data=df_cal)

        centers = [[10.0, 100.0, 20.0], [50.0, 500.0, 100.0], [90.0, 900.0, 180.0]]
        stds = [[3.0, 25.0, 5.0], [5.0, 40.0, 10.0], [4.0, 35.0, 8.0]]
        X_stat, _ = make_blobs(
            n_samples=120,
            centers=centers,
            cluster_std=stds,
            random_state=777,
        )
        stationary_window = pd.DataFrame(X_stat, columns=df_train.columns)

        drift_diag = engine.monitor_drift(stationary_window)

        assert set(drift_diag.keys()) == {
            "ks_statistic",
            "ks_pvalue",
            "drift_detected",
            "adapted",
            "observed_alert_rate",
        }
        assert isinstance(drift_diag["ks_pvalue"], float)
        assert isinstance(drift_diag["drift_detected"], bool)
        assert isinstance(drift_diag["adapted"], bool)
        assert drift_diag["drift_detected"] is False
        assert drift_diag["adapted"] is False

    def test_monitor_drift_triggers_autonomous_adaptation(
        self,
        continuous_tabular_splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies that significant distributional shift triggers drift alert and auto_adapt refit.
        """
        df_train, df_cal, _ = continuous_tabular_splits
        engine = PatternDiscoveryEngine(alpha=0.05, auto_adapt=True)
        engine.fit(df_train, cal_data=df_cal)

        shifted_window = pd.DataFrame(
            rng.normal(100.0, 1.0, size=(100, 3)),
            columns=df_train.columns,
        )

        drift_diag = engine.monitor_drift(shifted_window)

        assert drift_diag["drift_detected"] is True
        assert drift_diag["adapted"] is True
        assert engine.is_fitted is True
