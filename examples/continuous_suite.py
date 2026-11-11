"""
GEOMETRIC MODELS SUITE AND CONFORMAL BONFERRONI ENSEMBLE.

Demonstrates decoupled evaluation of four geometric anomaly scoring engines
(HDBSCAN, Isolation Forest, Local Outlier Factor, and One-Class SVM) on continuous
tabular spaces, coupled with inductive Bonferroni conformal calibration.

Evaluated workflow:
1. Synthetic multimodal Gaussian stream generation without data leakage.
2. Unsupervised fitting of raw geometric non-conformity scoring engines.
3. Raw discrimination profiling (ROC-AUC and Average Precision).
4. Inductive conformal calibration with two-sided Bonferroni bounds (alpha = 0.05).
5. Tripartite selective classification: 'anomaly', 'review', and 'no_alert'.
6. Empirical audit of population false positive rate control and coverage guarantees.
"""

from typing import Tuple
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import average_precision_score, roc_auc_score

from pde.calibration.selective import SelectiveClassifier
from pde.geometry.ensemble import GeometricBonferroniEnsemble
from pde.geometry.models import GeometricModelsSuite


def generate_synthetic_data(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates partitioned multimodal Gaussian continuous datasets across splits.

    Simulates normal observations drawn from three isotropic clusters alongside
    extreme out-of-distribution anomaly clusters.

    Args:
        seed (int, optional): Random seed for reproducible generation. Defaults
            to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - **X_train**: 360 normal training observations.
            - **X_cal**: 1000 independent normal calibration observations.
            - **X_test_norm**: 90 in-distribution normal test observations.
            - **X_anom**: 45 out-of-distribution anomaly observations.
    """
    centers = [[10.0, 100.0, 20.0], [50.0, 500.0, 100.0], [90.0, 900.0, 180.0]]
    stds = [[3.0, 25.0, 5.0], [5.0, 40.0, 10.0], [4.0, 35.0, 8.0]]

    total_normal = 360 + 1000 + 90
    X_normal, _ = make_blobs(
        n_samples=total_normal,
        centers=centers,
        cluster_std=stds,
        random_state=seed,
    )

    X_train = X_normal[:360]
    X_cal = X_normal[360:1360]
    X_test_norm = X_normal[1360:]

    rng = np.random.default_rng(seed + 999)
    X_anom = np.column_stack(
        [
            rng.uniform(130.0, 180.0, 45),
            rng.uniform(1300.0, 1800.0, 45),
            rng.uniform(260.0, 360.0, 45),
        ]
    )

    return X_train, X_cal, X_test_norm, X_anom


def main() -> None:
    """
    Executes the continuous geometric models suite and conformal ensemble workflow.
    """
    print("Geometric Models Suite And Conformal Bonferroni Ensemble")
    print("=" * 56)

    print("\n[INFO] Generating Multimodal Continuous Partitions Across Splits...")
    X_train, X_cal, X_test_norm, X_anom = generate_synthetic_data(seed=42)
    X_test = np.vstack([X_test_norm, X_anom])
    y_test = np.concatenate(
        [np.zeros(len(X_test_norm), dtype=int), np.ones(len(X_anom), dtype=int)]
    )

    print(f"\t• Training partition:    {X_train.shape[0]} normal samples")
    print(f"\t• Calibration partition: {X_cal.shape[0]} normal samples")
    print(
        f"\t• Test partition:        {X_test.shape[0]} samples "
        f"({len(X_test_norm)} normal, {len(X_anom)} anomalous)"
    )

    print("\n[INFO] Fitting Geometric Models Suite (HDBSCAN, IF, LOF, OCSVM)...")
    suite = GeometricModelsSuite(random_state=42)
    suite.fit(X_train)

    raw_scores = suite.compute_raw_scores(X_test)
    for model_name, scores in raw_scores.items():
        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        print(f"\t• Raw Score [{model_name:16s}]: ROC-AUC = {auc:.4f} | AP = {ap:.4f}")

    print("\n[INFO] Calibrating Inductive Conformal Bonferroni Ensemble (alpha = 0.05)...")
    ensemble = GeometricBonferroniEnsemble(random_state=42)
    ensemble.fit(X_train, X_cal)

    pvalues_dict = ensemble.predict_pvalues(X_test)
    classifier = SelectiveClassifier(alpha=0.05, review_boundary=0.20)

    print("\n[INFO] Selective Conformal Decision Performance And False Positive Rate Control:")
    print("-" * 76)
    print(
        f"{'Evaluated Method':24s} | {'ROC-AUC':8s} | {'AP':8s} | "
        f"{'Pop FPR':8s} | {'Op Recall':9s} | {'Coverage':8s}"
    )
    print("-" * 76)

    for method_name, p_vals in pvalues_dict.items():
        decisions = classifier.decide(p_vals)
        metrics = classifier.evaluate_metrics(y_test, decisions)
        rank_score = -p_vals
        auc = roc_auc_score(y_test, rank_score)
        ap = average_precision_score(y_test, rank_score)

        print(
            f"{method_name:24s} | {auc:.4f}   | {ap:.4f}   | "
            f"{metrics['population_fpr']:.4f}   | {metrics['operational_detection_rate']:.4f}    | "
            f"{metrics['coverage']:.4f}"
        )

    print("-" * 76)

    ens_decisions = classifier.decide(pvalues_dict["Bonferroni Ensemble"])
    ens_metrics = classifier.evaluate_metrics(y_test, ens_decisions)

    print("\n[INFO] Bonferroni Ensemble Calibration Guarantees Diagnostic:")
    print(
        f"\t• Observed false positive rate (FPR): {ens_metrics['population_fpr'] * 100:.2f}% "
        "(Nominal Bound <= 5.0%)"
    )
    print(
        f"\t• Operational detection recall:       {ens_metrics['operational_detection_rate'] * 100:.2f}%"
    )
    print(
        f"\t• Automated coverage (unreviewed):    {ens_metrics['coverage'] * 100:.2f}%"
    )
    print(
        f"\t• Referral rate to human review:      {ens_metrics['review_rate'] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
