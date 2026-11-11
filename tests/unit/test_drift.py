
from typing import Dict, Sequence, Tuple
import numpy as np
import pytest

import pde.drift
from pde.drift import DistributionDriftMonitor


@pytest.mark.unit
class TestDriftModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.drift.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains DistributionDriftMonitor.
        """
        expected = ["DistributionDriftMonitor"]
        assert sorted(pde.drift.__all__) == sorted(expected)
        assert pde.drift.DistributionDriftMonitor is DistributionDriftMonitor


@pytest.mark.unit
class TestDistributionDriftMonitor:
    """
    Unit test suite for DistributionDriftMonitor uniformity tests, thresholding, and window evaluation.
    """

    def test_init_default_and_custom_alpha(self) -> None:
        """
        Verifies default and custom alpha_drift initialization.
        """
        monitor_default = DistributionDriftMonitor()
        assert monitor_default.alpha_drift == 0.01

        monitor_custom = DistributionDriftMonitor(alpha_drift=0.05)
        assert monitor_custom.alpha_drift == 0.05

    def test_test_pvalue_uniformity_skips_small_samples(self) -> None:
        """
        Verifies that sample sizes smaller than 10 bypass the KS test to prevent false positives.
        """
        monitor = DistributionDriftMonitor(alpha_drift=0.01)

        stat, pval, drift_flag = monitor.test_pvalue_uniformity([])
        assert stat == 0.0
        assert pval == 1.0
        assert drift_flag is False

        small_sample = [0.001, 0.002, 0.001, 0.003]
        stat_s, pval_s, drift_s = monitor.test_pvalue_uniformity(small_sample)
        assert stat_s == 0.0
        assert pval_s == 1.0
        assert drift_s is False

        sample_9 = [0.01] * 9
        _, pval_9, drift_9 = monitor.test_pvalue_uniformity(sample_9)
        assert pval_9 == 1.0
        assert drift_9 is False

    def test_test_pvalue_uniformity_stationary_uniform_stream(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies that conformal p-values conforming to U(0, 1) do not trigger drift alerts.
        """
        monitor = DistributionDriftMonitor(alpha_drift=0.01)
        uniform_pvals = rng.uniform(0.0, 1.0, size=200)

        stat, pval, drift_flag = monitor.test_pvalue_uniformity(uniform_pvals)

        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert isinstance(drift_flag, bool)
        assert pval > monitor.alpha_drift
        assert drift_flag is False

    def test_test_pvalue_uniformity_detects_severe_drift(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies that non-uniform p-values strongly skewed towards zero trigger a drift alert.
        """
        monitor = DistributionDriftMonitor(alpha_drift=0.01)
        drifted_pvals = rng.beta(0.1, 10.0, size=150)

        stat, pval, drift_flag = monitor.test_pvalue_uniformity(drifted_pvals)

        assert stat > 0.30
        assert pval < monitor.alpha_drift
        assert drift_flag is True

    def test_evaluate_window_small_sample(self) -> None:
        """
        Verifies evaluate_window metrics when sample size is under the 10-observation threshold.
        """
        monitor = DistributionDriftMonitor(alpha_drift=0.01)
        p_vals = [0.01, 0.02, 0.50, 0.80]

        res = monitor.evaluate_window(p_vals, nominal_alpha=0.05)

        assert res["ks_statistic"] == 0.0
        assert res["ks_pvalue"] == 1.0
        assert res["drift_detected"] == 0.0
        assert np.isclose(res["observed_alert_rate"], 0.50)

    def test_evaluate_window_stationary_regime(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies evaluate_window returns expected diagnostic metrics under stationary operations.
        """
        monitor = DistributionDriftMonitor(alpha_drift=0.01)
        stationary_pvals = rng.uniform(0.0, 1.0, size=300)

        res = monitor.evaluate_window(stationary_pvals, nominal_alpha=0.05)

        assert set(res.keys()) == {
            "ks_statistic",
            "ks_pvalue",
            "drift_detected",
            "observed_alert_rate",
        }
        assert res["drift_detected"] == 0.0
        assert res["ks_pvalue"] > monitor.alpha_drift
        assert np.isclose(res["observed_alert_rate"], 0.05, atol=0.04)

    def test_evaluate_window_drifted_regime(
        self,
        rng: np.random.Generator,
    ) -> None:
        """
        Verifies evaluate_window alerts and reports elevated false positive rate under drift.
        """
        monitor = DistributionDriftMonitor(alpha_drift=0.01)
        drifted_pvals = rng.uniform(0.0, 0.04, size=100)

        res = monitor.evaluate_window(drifted_pvals, nominal_alpha=0.05)

        assert res["drift_detected"] == 1.0
        assert res["ks_pvalue"] < monitor.alpha_drift
        assert np.isclose(res["observed_alert_rate"], 1.0)
