
from typing import Dict, Sequence, Tuple
import numpy as np
from scipy.stats import kstest


class DistributionDriftMonitor:
    """
    Monitors inference windows to detect distributional drift against U(0, 1).

    Evaluates the uniformity of conformal p-values over sliding or batch windows
    using the Kolmogorov-Smirnov test, alerting when observed distributions
    significantly deviate from theoretical standard uniformity.

    Attributes:
        alpha_drift (float): Significance level threshold for rejecting the null
            hypothesis of uniform p-values (drift detection).
    """

    def __init__(self, alpha_drift: float = 0.01):
        """
        Initializes the DistributionDriftMonitor instance.

        Args:
            alpha_drift (float, optional): Significance level threshold for the
                Kolmogorov-Smirnov uniformity test. Defaults to 0.01.
        """
        self.alpha_drift = float(alpha_drift)


    def test_pvalue_uniformity(self, p_values: Sequence[float]) -> Tuple[float, float, bool]:
        """
        Runs the Kolmogorov-Smirnov goodness-of-fit test against U(0, 1).

        Compares the empirical distribution of conformal p-values to a continuous
        uniform distribution on [0, 1]. For samples smaller than 10 observations,
        the test is skipped to prevent false positives from low statistical power.

        Args:
            p_values (Sequence[float]): Sequence of observed conformal p-values
                to evaluate.

        Returns:
            Tuple[float, float, bool]: A tuple containing:
                - **ks_statistic** (`float`): The two-sided Kolmogorov-Smirnov test
                  statistic measuring maximum divergence.
                - **ks_pvalue** (`float`): Two-sided p-value of the Kolmogorov-Smirnov
                  test.
                - **drift_detected** (`bool`): True if `ks_pvalue < alpha_drift`,
                  indicating significant distributional deviation from U(0, 1).
        """
        p_vals = np.asarray(p_values, dtype=float)

        if len(p_vals) < 10:
            return 0.0, 1.0, False

        stat, pval = kstest(p_vals, "uniform")
        drift_flag = bool(pval < self.alpha_drift)

        return float(stat), float(pval), drift_flag


    def evaluate_window(self, p_values: Sequence[float], nominal_alpha: float = 0.05) -> Dict[str, float]:
        """
        Evaluates drift status and computes the empirical alert rate over a window.

        Conducts the KS uniformity test and calculates the proportion of p-values
        that fall below the nominal significance threshold.

        Args:
            p_values (Sequence[float]): Sequence of observed conformal p-values
                in the current window.
            nominal_alpha (float, optional): Nominal target significance level
                used to compute the observed alert rate. Defaults to 0.05.

        Returns:
            Dict[str, float]: Dictionary containing computed window metrics:
                - `'ks_statistic'`: Maximum distance between the empirical and
                  uniform cumulative distributions.
                - `'ks_pvalue'`: P-value corresponding to the KS test.
                - `'drift_detected'`: Binary flag (`1.0` if drift is detected,
                  `0.0` otherwise).
                - `'observed_alert_rate'`: Empirical proportion of observations
                  with `p <= nominal_alpha`.
        """
        p_vals = np.asarray(p_values, dtype=float)
        stat, pval, drift_flag = self.test_pvalue_uniformity(p_vals)
        observed_fpr = float(np.mean(p_vals <= nominal_alpha))

        return {
            "ks_statistic": stat,
            "ks_pvalue": pval,
            "drift_detected": float(drift_flag),
            "observed_alert_rate": observed_fpr,
        }