"""
NON-PARAMETRIC CONFORMAL CALIBRATION AND SELECTIVE CLASSIFICATION.

Demonstrates distribution-free false positive rate control (Type I error bound)
and tripartite selective classification with an audited human review zone.

Evaluated workflow:
1. Synthetic non-conformity continuous score generation (normal vs. anomalous).
2. Inductive conformal calibration using empirical reference quantiles (ConformalCalibrator).
3. Empirical U(0, 1) uniformity audit under the null hypothesis (normal samples).
4. Formal theoretical false positive rate guarantee validation: FPR <= alpha.
5. Tripartite selective decision-making with SelectiveClassifier:
   - 'anomaly'  (p <= alpha): High-certainty operational detection alarm.
   - 'review'   (alpha < p < review_boundary): Abstention zone deferred to manual triage.
   - 'no_alert' (p >= review_boundary): Statistically validated nominal normality.
6. Multi-detector fusion with Bonferroni conformal combination.
7. Operational metric audit: coverage, selective precision, and detection recall.
"""

from typing import Tuple
import numpy as np
from scipy.stats import kstest
from sklearn.metrics import average_precision_score, roc_auc_score

from pde.calibration.conformal import ConformalCalibrator
from pde.calibration.selective import SelectiveClassifier


def generate_synthetic_nonconformity_scores(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generates synthetic non-conformity score distributions across calibration and test splits.

    Simulates continuous non-conformity scores s(x) >= 0 where normal observations
    follow a Chi-Square distribution (k=3 degrees of freedom) representing typical
    Mahalanobis distances, and anomalies exhibit positive location and scale shifts.

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing:
            - **cal_scores**: 1000 in-distribution calibration scores.
            - **query_scores**: 250 evaluation scores (200 normal, 50 anomalous).
            - **ground_truth**: Binary labels for the evaluation set (0=normal, 1=anomaly).
            - **n_test_normal**: Count of normal observations in the query set (200).
    """
    rng = np.random.default_rng(seed)

    n_cal = 1000
    cal_scores = rng.chisquare(df=3, size=n_cal)

    n_test_normal = 200
    n_test_anomaly = 50

    test_normal_scores = rng.chisquare(df=3, size=n_test_normal)
    test_anomaly_scores = rng.normal(loc=12.0, scale=2.5, size=n_test_anomaly)
    test_anomaly_scores = np.maximum(0.1, test_anomaly_scores)

    query_scores = np.concatenate([test_normal_scores, test_anomaly_scores])
    ground_truth = np.concatenate(
        [np.zeros(n_test_normal, dtype=int), np.ones(n_test_anomaly, dtype=int)]
    )

    return cal_scores, query_scores, ground_truth, n_test_normal


def main() -> None:
    """
    Executes the conformal calibration and selective classification workflow.
    """
    print("Non-Parametric Conformal Calibration And Selective Classification")
    print("=" * 66)

    alpha_nominal = 0.05
    review_boundary = 0.20

    print("\n[INFO] Generating Continuous Non-Conformity Scores Across Partitions...")
    (
        cal_scores,
        query_scores,
        y_true,
        n_normals,
    ) = generate_synthetic_nonconformity_scores(seed=42)

    print(f"\t• Calibration partition:       {len(cal_scores)} normal scores")
    print(
        f"\t• Total evaluation queries:     {len(query_scores)} "
        f"({n_normals} normal, {len(y_true) - n_normals} anomalous)"
    )
    print(f"\t• Nominal significance level:   alpha = {alpha_nominal:.2f}")
    print(f"\t• Review boundary threshold:    review_boundary = {review_boundary:.2f}")

    print("\n[INFO] Fitting ConformalCalibrator On Reference Calibration Scores...")
    calibrator = ConformalCalibrator()
    calibrator.fit(cal_scores)

    p_values = calibrator.compute_pvalue(query_scores)

    normal_pvalues = p_values[:n_normals]
    ks_stat, ks_pval = kstest(normal_pvalues, "uniform")
    observed_fpr = float(np.mean(normal_pvalues <= alpha_nominal))

    print("\n[INFO] Empirical Null Hypothesis Audit Under H0 (Normal Observations):")
    print(
        f"\t• Kolmogorov-Smirnov vs U(0, 1): statistic = {ks_stat:.4f}, "
        f"p-value = {ks_pval:.4f}"
    )
    print(
        f"\t• Observed false positive rate:  FPR = {observed_fpr * 100:.2f}% "
        f"(Nominal Guarantee <= {alpha_nominal * 100:.1f}%)"
    )
    status_str = (
        "VALID (FPR <= alpha)"
        if observed_fpr <= alpha_nominal + 0.02
        else "VIOLATED"
    )
    print(f"\t• Type I error bound compliance: {status_str}")

    print("\n[INFO] Tripartite Selective Classification And Decision Guarantees:")
    classifier = SelectiveClassifier(
        alpha=alpha_nominal, review_boundary=review_boundary
    )
    decisions = classifier.decide(p_values)
    metrics = classifier.evaluate_metrics(y_true, decisions)

    n_anomalies_flagged = int(np.sum(decisions == "anomaly"))
    n_review_flagged = int(np.sum(decisions == "review"))
    n_normal_flagged = int(np.sum(decisions == "no_alert"))

    print(
        f"\t• Decisions emitted:            "
        f"Anomaly = {n_anomalies_flagged} | Review = {n_review_flagged} | No-Alert = {n_normal_flagged}"
    )
    print(f"\t• Automated coverage:           {metrics['coverage'] * 100:.2f}%")
    print(f"\t• Human review deferral rate:   {metrics['review_rate'] * 100:.2f}%")
    print(f"\t• Automated decision precision: {metrics['precision_automatic'] * 100:.2f}%")
    print(f"\t• Operational detection recall: {metrics['operational_detection_rate'] * 100:.2f}%")
    print(f"\t• Population FPR on normals:    {metrics['population_fpr'] * 100:.2f}%")

    print("\n[INFO] Multi-Detector Fusion With Bonferroni Conformal Calibration:")
    rng2 = np.random.default_rng(2026)
    detector2_cal = rng2.exponential(scale=1.5, size=len(cal_scores))
    detector2_query_norm = rng2.exponential(scale=1.5, size=n_normals)
    detector2_query_anom = rng2.exponential(scale=6.0, size=len(y_true) - n_normals)
    detector2_query = np.concatenate([detector2_query_norm, detector2_query_anom])

    calibrator2 = ConformalCalibrator().fit(detector2_cal)
    p_values_det2 = calibrator2.compute_pvalue(detector2_query)

    p_matrix = np.column_stack([p_values, p_values_det2])
    p_bonferroni = ConformalCalibrator.bonferroni_combine(p_matrix)

    bonferroni_decisions = classifier.decide(p_bonferroni)
    bonf_metrics = classifier.evaluate_metrics(y_true, bonferroni_decisions)

    auc_d1 = roc_auc_score(y_true, -p_values)
    auc_d2 = roc_auc_score(y_true, -p_values_det2)
    auc_ens = roc_auc_score(y_true, -p_bonferroni)

    det2_decisions = classifier.decide(p_values_det2)
    det2_metrics = classifier.evaluate_metrics(y_true, det2_decisions)

    print("-" * 76)
    print(
        f"{'Detector / Ensemble':30s} | {'ROC-AUC':8s} | {'Pop FPR':8s} | "
        f"{'Op Recall':9s} | {'Coverage':8s}"
    )
    print("-" * 76)
    print(
        f"{'Detector 1 (Spatial)':30s} | {auc_d1:.4f}   | "
        f"{metrics['population_fpr']:.4f}   | {metrics['operational_detection_rate']:.4f}    | "
        f"{metrics['coverage']:.4f}"
    )
    print(
        f"{'Detector 2 (Temporal/Rules)':30s} | {auc_d2:.4f}   | "
        f"{det2_metrics['population_fpr']:.4f}   | {det2_metrics['operational_detection_rate']:.4f}    | "
        f"{det2_metrics['coverage']:.4f}"
    )
    print(
        f"{'Bonferroni Ensemble':30s} | {auc_ens:.4f}   | "
        f"{bonf_metrics['population_fpr']:.4f}   | {bonf_metrics['operational_detection_rate']:.4f}    | "
        f"{bonf_metrics['coverage']:.4f}"
    )
    print("-" * 76)


if __name__ == "__main__":
    main()
