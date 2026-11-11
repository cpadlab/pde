"""
DISTRIBUTION DRIFT MONITORING AND AUTONOMOUS RE-CALIBRATION.

Demonstrates non-parametric detection of distribution drift using Kolmogorov-Smirnov
goodness-of-fit tests on conformal p-values, alongside unsupervised autonomous
re-calibration upon drift alerts.

Evaluated workflow:
1. Streaming trajectory generation spanning three regimes:
   - Stationary baseline regime (pre-drift).
   - Mean-shifted operational regime (active drift).
   - True anomalies arising within the newly shifted regime.
2. Continuous conformal p-value monitoring against U(0, 1) using DistributionDriftMonitor.
3. Obsolete frozen model audit: false positive rate explosion (FPR ~ 30-40%).
4. Formal statistical drift detection via Kolmogorov-Smirnov test (p_KS < alpha_drift = 0.01).
5. Autonomous adaptation: reference parameter re-estimation and conformal re-calibration.
6. Restoration of false positive rate control (FPR ~ 5%) and anomaly detection power.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pde.calibration.conformal import ConformalCalibrator
from pde.calibration.selective import SelectiveClassifier
from pde.drift.monitor import DistributionDriftMonitor


def generate_streaming_drift_data(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates multi-phase streaming datasets simulating distribution drift.

    Simulates an operational environment undergoing three sequential states:
    a stationary baseline, an in-stream mean shift across dimensions 0 and 1,
    and extreme out-of-distribution anomalies arising inside the shifted regime.

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - **train_data**: Baseline training observations (360 samples, 3 features).
            - **cal_data**: Independent baseline calibration observations (1000 samples).
            - **pre_drift_stream**: In-distribution stationary streaming window (100 samples).
            - **drift_stream**: Shifted streaming window (200 samples).
            - **new_regime_anomalies**: True anomalous instances in shifted space (45 samples).
    """
    rng = np.random.default_rng(seed)

    train_data = rng.normal(loc=0.0, scale=1.0, size=(360, 3))
    cal_data = rng.normal(loc=0.0, scale=1.0, size=(1000, 3))
    pre_drift_stream = rng.normal(loc=0.0, scale=1.0, size=(100, 3))

    drift_stream = rng.normal(loc=0.0, scale=1.0, size=(200, 3))
    drift_stream[:, 0] += 1.50
    drift_stream[:, 1] += 1.20

    rng_anom = np.random.default_rng(seed + 888)
    new_regime_anomalies = rng_anom.uniform(6.5, 11.0, size=(45, 3))

    return (
        train_data,
        cal_data,
        pre_drift_stream,
        drift_stream,
        new_regime_anomalies,
    )


def compute_standardized_scores(
    data: np.ndarray, mu: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """
    Computes multidimensional standardized Euclidean distances from reference parameters.

    Args:
        data (np.ndarray): Input feature matrix of shape (n_samples, n_features).
        mu (np.ndarray): Feature-wise empirical mean vector.
        std (np.ndarray): Feature-wise empirical standard deviation vector.

    Returns:
        np.ndarray: Vector of Euclidean norms of standardized scores.
    """
    z = (data - mu) / std
    return np.linalg.norm(z, axis=1)


def main() -> None:
    """
    Executes the distribution drift monitoring and autonomous re-calibration workflow.
    """
    print("Distribution Drift Monitoring And Autonomous Re-Calibration")
    print("=" * 60)

    alpha_nominal = 0.05
    alpha_drift = 0.01

    print("\n[INFO] Generating Multi-Phase Streaming Data With Distribution Shift...")
    (
        train,
        cal,
        stream_pre,
        stream_drift,
        anom_new,
    ) = generate_streaming_drift_data(seed=42)

    print(f"\t• Baseline training samples:    {train.shape[0]}")
    print(f"\t• Baseline calibration samples: {cal.shape[0]}")
    print(f"\t• Pre-drift stationary stream:  {stream_pre.shape[0]}")
    print(f"\t• Shifted operational stream:   {stream_drift.shape[0]}")
    print(f"\t• New regime ground anomalies:  {anom_new.shape[0]}")

    print("\n[INFO] Establishing Baseline Parameters And Conformal Calibration...")
    mu_base = np.mean(train, axis=0)
    std_base = np.maximum(1e-6, np.std(train, axis=0))

    cal_scores_base = compute_standardized_scores(cal, mu_base, std_base)
    calibrator = ConformalCalibrator().fit(cal_scores_base)
    classifier = SelectiveClassifier(alpha=alpha_nominal, review_boundary=0.20)
    drift_monitor = DistributionDriftMonitor(alpha_drift=alpha_drift)

    print("\n[INFO] Monitoring Pre-Drift Stationary Window (Normal Regime)...")
    s_pre = compute_standardized_scores(stream_pre, mu_base, std_base)
    p_pre = calibrator.compute_pvalue(s_pre)

    diag_pre = drift_monitor.evaluate_window(p_pre, nominal_alpha=alpha_nominal)
    print(
        f"\t• KS test vs U(0, 1): p-value = {diag_pre['ks_pvalue']:.4f} "
        f"(statistic = {diag_pre['ks_statistic']:.4f})"
    )
    print(f"\t• Drift alert triggered:        {bool(diag_pre['drift_detected'])}")
    print(
        f"\t• Observed false positive rate: {diag_pre['observed_alert_rate'] * 100:.2f}% "
        f"(Nominal = {alpha_nominal * 100:.1f}%)"
    )

    print("\n[INFO] Evaluating Shifted Stream Under Frozen Baseline Model...")
    s_drift_frozen = compute_standardized_scores(stream_drift, mu_base, std_base)
    p_drift_frozen = calibrator.compute_pvalue(s_drift_frozen)

    diag_frozen = drift_monitor.evaluate_window(
        p_drift_frozen, nominal_alpha=alpha_nominal
    )
    print(
        f"\t• KS test vs U(0, 1): p-value = {diag_frozen['ks_pvalue']:.2e} "
        f"(statistic = {diag_frozen['ks_statistic']:.4f})"
    )
    print(
        f"\t• Drift alert triggered:        {bool(diag_frozen['drift_detected'])} "
        f"(p_KS < {alpha_drift})"
    )
    print(
        f"\t• Observed false positive rate: {diag_frozen['observed_alert_rate'] * 100:.2f}% "
        "(Model Degradation)"
    )

    print("\n[INFO] Executing Autonomous Re-Calibration Over Shifted Influx Buffer...")
    adapt_buffer = stream_drift[:100]
    stream_post_adapt = stream_drift[100:]

    mu_adapted = np.mean(adapt_buffer, axis=0)
    std_adapted = np.maximum(1e-6, np.std(adapt_buffer, axis=0))

    cal_scores_adapted = compute_standardized_scores(
        adapt_buffer, mu_adapted, std_adapted
    )
    calibrator_adapted = ConformalCalibrator().fit(cal_scores_adapted)

    print("\n[INFO] Verifying System Stability Following Autonomous Adaptation...")
    s_post = compute_standardized_scores(stream_post_adapt, mu_adapted, std_adapted)
    p_post = calibrator_adapted.compute_pvalue(s_post)

    diag_post = drift_monitor.evaluate_window(p_post, nominal_alpha=alpha_nominal)
    print(
        f"\t• KS test vs U(0, 1): p-value = {diag_post['ks_pvalue']:.4f} "
        f"(statistic = {diag_post['ks_statistic']:.4f})"
    )
    print(f"\t• Drift alert triggered:        {bool(diag_post['drift_detected'])}")
    print(
        f"\t• Observed false positive rate: {diag_post['observed_alert_rate'] * 100:.2f}% "
        "(Restored to Nominal)"
    )

    print("\n[INFO] Anomaly Discrimination Performance In Re-Calibrated Regime:")
    s_anom = compute_standardized_scores(anom_new, mu_adapted, std_adapted)
    p_anom = calibrator_adapted.compute_pvalue(s_anom)

    eval_scores = np.concatenate([s_post, s_anom])
    eval_pvals = np.concatenate([p_post, p_anom])
    eval_labels = np.concatenate(
        [np.zeros(len(s_post), dtype=int), np.ones(len(anom_new), dtype=int)]
    )

    auc_new = roc_auc_score(eval_labels, eval_scores)
    ap_new = average_precision_score(eval_labels, eval_scores)
    anom_decisions = classifier.decide(eval_pvals)
    op_recall = np.mean(anom_decisions[eval_labels == 1] != "no_alert")

    op_recall_str = f"{op_recall * 100:.2f}%"
    post_fpr_str = f"{diag_post['observed_alert_rate'] * 100:.2f}%"

    print("-" * 72)
    print(
        f"{'Evaluated Metric (Adapted Regime)':36s} | {'Observed':14s} | {'Target Bound':14s}"
    )
    print("-" * 72)
    print(f"{'ROC-AUC (New Anomalies)':36s} | {auc_new:<14.4f} | >= 0.9500")
    print(f"{'Average Precision (AP)':36s} | {ap_new:<14.4f} | >= 0.9500")
    print(
        f"{'Operational Detection Recall':36s} | {op_recall_str:<14s} | 100.00%"
    )
    print(
        f"{'Post-Adaptation FPR':36s} | {post_fpr_str:<14s} | <= 5.00%"
    )
    print("-" * 72)

    print("\n[INFO] Drift Lifecycle Diagnostic Summary:")
    print(
        f"\t• Pre-drift baseline:   FPR = {diag_pre['observed_alert_rate'] * 100:5.2f}% | "
        f"Drift alert = {bool(diag_pre['drift_detected'])}"
    )
    print(
        f"\t• Frozen model drift:   FPR = {diag_frozen['observed_alert_rate'] * 100:5.2f}% | "
        f"Drift alert = {bool(diag_frozen['drift_detected'])} (Alert)"
    )
    print(
        f"\t• Post-adaptation:      FPR = {diag_post['observed_alert_rate'] * 100:5.2f}% | "
        f"Drift alert = {bool(diag_post['drift_detected'])} (Nominal)"
    )


if __name__ == "__main__":
    main()
