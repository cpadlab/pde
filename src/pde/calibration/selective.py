
from typing import Union, Sequence, Dict
import numpy as np


class SelectiveClassifier:
    """
    Classifies observations based on their conformal certainty level.

    Applies selective classification thresholds to conformal p-values,
    separating definite anomalies, uncertain samples requiring manual review,
    and regular non-alerts.

    Attributes:
        alpha (float): Significance level threshold below which observations
            are classified as anomalies.
        review_boundary (float): Upper p-value threshold defining the boundary
            of the manual review region.
    """


    def __init__(self, alpha: float = 0.05, review_boundary: float = 0.20):
        """
        Initializes the SelectiveClassifier with decision thresholds.

        Args:
            alpha (float, optional): Critical threshold for anomaly confirmation.
                Defaults to 0.05.
            review_boundary (float, optional): Upper bound for ambiguous samples
                subject to manual review. Defaults to 0.20.

        Raises:
            ValueError: If the condition `0 < alpha < review_boundary < 1`
                is not satisfied.
        """
        if not (0.0 < alpha < review_boundary < 1.0):
            raise ValueError(f"The following must hold: 0 < alpha ({alpha}) < review_boundary ({review_boundary}) < 1")
    
        self.alpha = float(alpha)
        self.review_boundary = float(review_boundary)


    def decide(self, p_values: Union[float, Sequence[float]]) -> np.ndarray:
        """
        Maps conformal p-values to operational decision labels.

        Classification rules:
            - `p <= alpha`: `'anomaly'` (high-risk confirmed alert).
            - `alpha < p < review_boundary`: `'review'` (ambiguous zone, requires manual check).
            - `p >= review_boundary`: `'no_alert'` (consistent with normal behavior).

        Args:
            p_values (Union[float, Sequence[float]]): Single p-value or sequence
                of p-values to categorize.

        Returns:
            np.ndarray: Array of string labels (`'anomaly'`, `'review'`, or
                `'no_alert'`) with the same shape as `p_values`.
        """
        p_vals = np.asarray(p_values, dtype=float)

        decisions = np.full(p_vals.shape, "no_alert", dtype=object)
        decisions[(p_vals > self.alpha) & (p_vals < self.review_boundary)] = "review"
        decisions[p_vals <= self.alpha] = "anomaly"

        return decisions


    def evaluate_metrics(self, ground_truth: Sequence[int], decisions: Sequence[str]) -> Dict[str, float]:
        """
        Calculates coverage, selective risk, and operational performance metrics.

        Computes automated vs. reviewed proportions, precision and recall on
        unambiguous decisions, full operational detection rates, and false
        positive rates.

        Args:
            ground_truth (Sequence[int]): Binary ground truth labels, where `1`
                represents an anomaly and `0` represents a normal sample.
            decisions (Sequence[str]): Operational decision labels corresponding
                to each sample (`'anomaly'`, `'review'`, or `'no_alert'`).

        Returns:
            Dict[str, float]: Dictionary containing computed metrics:
                - `'coverage'`: Proportion of instances resolved automatically.
                - `'review_rate'`: Proportion of instances routed to manual review.
                - `'precision_automatic'`: Precision of automatic anomaly alerts.
                - `'recall_automatic'`: Recall of anomalies captured without manual review.
                - `'operational_detection_rate'`: Recall including both automated
                  alerts and reviewed anomalies.
                - `'population_fpr'`: False positive rate over all normal samples.
        """
        y_true = np.asarray(ground_truth, dtype=int)
        y_dec = np.asarray(decisions)

        automatic_mask = y_dec != "review"
        total_samples = len(y_true)
        coverage = float(np.mean(automatic_mask))

        tp = np.sum((y_true == 1) & (y_dec == "anomaly"))
        fp = np.sum((y_true == 0) & (y_dec == "anomaly"))
        fn = np.sum((y_true == 1) & (y_dec == "no_alert"))
        tn = np.sum((y_true == 0) & (y_dec == "no_alert"))
        reviewed_anoms = np.sum((y_true == 1) & (y_dec == "review"))

        total_anomalies = np.sum(y_true == 1)
        total_normals = np.sum(y_true == 0)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall_auto = float(tp / total_anomalies) if total_anomalies > 0 else 0.0
        op_detection = float((tp + reviewed_anoms) / total_anomalies) if total_anomalies > 0 else 0.0
        pop_fpr = float(fp / total_normals) if total_normals > 0 else 0.0

        return {
            "coverage": coverage,
            "review_rate": 1.0 - coverage,
            "precision_automatic": precision,
            "recall_automatic": recall_auto,
            "operational_detection_rate": op_detection,
            "population_fpr": pop_fpr,
        }