
from typing import Union, Sequence
import numpy as np


class ConformalCalibrator:
    """
    Inductive Conformal Calibrator based on empirical non-conformity quantiles.

    Computes exact conformal p-values from reference calibration scores and
    combines statistical evidence across multiple independent detectors.

    Attributes:
        reference_scores_ (np.ndarray): Calibration scores sorted in ascending
            order. Empty until `fit` is called.
        is_calibrated_ (bool): Flag indicating whether the calibrator has been
            fitted with reference scores.
    """


    def __init__(self):
        """
        Initializes an uncalibrated ConformalCalibrator instance.
        """
        self.reference_scores_: np.ndarray = np.array([], dtype=float)
        self.is_calibrated_: bool = False


    def fit(self, reference_scores: Sequence[float]) -> "ConformalCalibrator":
        """
        Fits the reference distribution exclusively using normal calibration scores.

        Sorts and stores non-conformity scores from the calibration dataset to
        enable empirical quantile calculations.

        Args:
            reference_scores (Sequence[float]): Non-conformity scores computed on
                the calibration set without data leakage.

        Returns:
            ConformalCalibrator: The fitted instance (`self`).

        Raises:
            ValueError: If `reference_scores` is empty.
        """
        scores = np.asarray(reference_scores, dtype=float)
        if len(scores) == 0:
            raise ValueError("The set of calibration scores cannot be empty.")

        self.reference_scores_ = np.sort(scores)
        self.is_calibrated_ = True

        return self


    def compute_pvalue(self, query_scores: Union[float, Sequence[float]]) -> np.ndarray:
        """
        Computes exact inductive conformal p-values for query scores.

        Applies the standard inductive conformal prediction formula:
            p(s) = (1 + sum(s_cal >= s)) / (N_cal + 1)

        Args:
            query_scores (Union[float, Sequence[float]]): Single score or sequence
                of query scores to evaluate against the calibration distribution.

        Returns:
            np.ndarray: 1D array of calculated conformal p-values.

        Raises:
            RuntimeError: If called before fitting the calibrator with `fit`.
        """
        if not self.is_calibrated_:
            raise RuntimeError("The calibrator must be adjusted using .fit() before calculating p-values.")

        query = np.atleast_1d(np.asarray(query_scores, dtype=float))
        n_ref = len(self.reference_scores_)

        idx = np.searchsorted(self.reference_scores_, query, side="left")
        counts_greater_equal = n_ref - idx
        p_values = (1.0 + counts_greater_equal) / (n_ref + 1.0)

        return p_values


    @staticmethod
    def bonferroni_combine(p_values_matrix: np.ndarray) -> np.ndarray:
        """
        Combines p-values from multiple detectors using Bonferroni correction.

        Calculates conservative ensemble p-values row-wise via:
            p_ens = min(1.0, M * min(p_1, ..., p_M))

        Args:
            p_values_matrix (np.ndarray): Array of p-values of shape
                `(n_samples, n_detectors)`. If a 1D array is provided, it is
                returned unchanged.

        Returns:
            np.ndarray: 1D array containing the combined, bounded p-values in [0.0, 1.0].
        """
        p_mat = np.asarray(p_values_matrix, dtype=float)
        if p_mat.ndim == 1:
            return p_mat
        
        m_detectors = p_mat.shape[1]
        min_p = np.min(p_mat, axis=1)
        
        return np.minimum(1.0, m_detectors * min_p)


    def to_dict(self) -> dict:
        """
        Serializes the calibrator state to a dictionary.

        Returns:
            dict: Dictionary containing `reference_scores` as a list of floats
                and `is_calibrated` as a boolean.
        """
        return {
            "reference_scores": self.reference_scores_.tolist(),
            "is_calibrated": self.is_calibrated_,
        }


    @classmethod
    def from_dict(cls, data: dict) -> "ConformalCalibrator":
        """
        Reconstructs a ConformalCalibrator instance from a serialized dictionary.

        Args:
            data (dict): Dictionary with keys `'reference_scores'` (sequence of
                numbers) and `'is_calibrated'` (boolean).

        Returns:
            ConformalCalibrator: A new reconstructed and calibrated instance.
        """
        inst = cls()
        inst.reference_scores_ = np.array(data["reference_scores"], dtype=float)
        inst.is_calibrated_ = bool(data["is_calibrated"])
        
        return inst