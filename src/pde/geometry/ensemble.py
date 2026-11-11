
from typing import Dict, Optional
import numpy as np

from pde.calibration.conformal import ConformalCalibrator
from pde.geometry.models import GeometricModelsSuite


class GeometricBonferroniEnsemble:
    """
    Orchestrates fitting, calibration, and conformal combination of the geometric quartet.

    Trains the underlying geometric detector suite on nominal training data,
    fits inductive conformal calibrators on held-out calibration scores, and
    derives conservative ensemble p-values using Bonferroni correction.

    Attributes:
        ALGORITHMS (List[str]): List of detector names included in the quartet:
            `["HDBSCAN", "Isolation Forest", "LOF", "One-Class SVM"]`.
        suite (GeometricModelsSuite): Anomaly detection models suite encapsulating
            the geometric estimators.
        calibrators_ (Dict[str, ConformalCalibrator]): Mapping from detector
            name to its corresponding fitted `ConformalCalibrator` instance.
        is_calibrated_ (bool): Flag indicating whether the detectors and their
            calibrators have been fitted.
    """


    ALGORITHMS = ["HDBSCAN", "Isolation Forest", "LOF", "One-Class SVM"]


    def __init__(self, random_state: Optional[int] = 42):
        """
        Initializes the GeometricBonferroniEnsemble instance.

        Args:
            random_state (Optional[int], optional): Random seed used to configure
                reproducible detectors within `GeometricModelsSuite`.
                Defaults to 42.
        """
        self.suite = GeometricModelsSuite(random_state=random_state)
        self.calibrators_: Dict[str, ConformalCalibrator] = {
            algo: ConformalCalibrator() for algo in self.ALGORITHMS
        }

        self.is_calibrated_: bool = False

    def fit(self, x_train: np.ndarray, x_calibration: np.ndarray) -> "GeometricBonferroniEnsemble":
        """
        Fits base detectors on training data and calibrates non-parametric quantiles.

        Trains the geometric models suite on nominal baseline observations,
        evaluates raw anomaly scores on the independent calibration partition,
        and fits empirical reference distributions for each calibrator.

        Args:
            x_train (np.ndarray): Training dataset of shape `(n_train_samples, n_features)`
                containing strictly normal/reference data.
            x_calibration (np.ndarray): Held-out calibration dataset of shape
                `(n_cal_samples, n_features)` without data leakage.

        Returns:
            GeometricBonferroniEnsemble: The fitted and calibrated ensemble instance (`self`).
        """
        self.suite.fit(x_train)

        cal_scores = self.suite.compute_raw_scores(x_calibration)

        for algo in self.ALGORITHMS:
            self.calibrators_[algo].fit(cal_scores[algo])

        self.is_calibrated_ = True

        return self

    def predict_pvalues(self, x_query: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculates individual conformal p-values and the aggregated Bonferroni ensemble.

        Computes raw non-conformity scores for query observations, transforms
        them into exact inductive conformal p-values for each base detector,
        and applies Bonferroni correction:
            p_ens = min(1.0, M * min(p_1, ..., p_M))

        Args:
            x_query (np.ndarray): Query observation array of shape
                `(n_query_samples, n_features)` to evaluate.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping each algorithm name and the key
                `'Bonferroni Ensemble'` to 1D arrays of computed p-values of shape
                `(n_query_samples,)`.

        Raises:
            RuntimeError: If called before fitting and calibrating via `fit`.
        """
        if not self.is_calibrated_:
            raise RuntimeError("The model must be calibrated using .fit() before making predictions.")

        query_scores = self.suite.compute_raw_scores(x_query)
        pvalues_dict = {}

        for algo in self.ALGORITHMS:
            pvalues_dict[algo] = self.calibrators_[algo].compute_pvalue(query_scores[algo])

        p_matrix = np.column_stack([pvalues_dict[algo] for algo in self.ALGORITHMS])
        pvalues_dict["Bonferroni Ensemble"] = ConformalCalibrator.bonferroni_combine(p_matrix)

        return pvalues_dict