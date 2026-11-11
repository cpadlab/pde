
from typing import Dict, Sequence
import numpy as np
from sklearn.cluster import KMeans


class CollectiveCompositionDetector:
    """
    Evaluates the compositional discrepancy of observation batches or groups.

    Fits a clustering model on baseline training data to establish an expected
    modal mixture distribution. Given a new group of samples, it quantifies
    distributional shifts across clusters using statistical divergence and
    goodness-of-fit metrics.

    Attributes:
        n_clusters (int): Number of clusters defining the categorical mixture.
        random_state (int): Random state seed used for reproducible k-means clustering.
        kmeans_ (Optional[KMeans]): Fitted k-means clustering estimator. None
            before `fit`.
        baseline_proportions_ (Optional[np.ndarray]): Empirical baseline cluster
            proportions of shape `(n_clusters,)`. None before `fit`.
        is_fitted_ (bool): Flag indicating whether the baseline distribution has
            been estimated.
    """


    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        """
        Initializes the CollectiveCompositionDetector instance.

        Args:
            n_clusters (int, optional): Number of clusters to segment the feature
                space into. Defaults to 3.
            random_state (int, optional): Random seed passed to the KMeans
                estimator. Defaults to 42.
        """
        self.n_clusters = int(n_clusters)
        self.random_state = random_state

        self.kmeans_: Optional[KMeans] = None
        self.baseline_proportions_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False


    def fit(self, X_train_scaled: np.ndarray) -> "CollectiveCompositionDetector":
        """
        Learns the baseline modal composition in the scaled training space.

        Fits a k-means clustering model on the scaled training features and
        computes the expected empirical cluster proportion vector.

        Args:
            X_train_scaled (np.ndarray): Scaled training data array of shape
                `(n_samples, n_features)`.

        Returns:
            CollectiveCompositionDetector: The fitted detector instance (`self`).
        """
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(X_train_scaled)

        train_labels = self.kmeans_.predict(X_train_scaled)
        counts = np.bincount(train_labels, minlength=self.n_clusters).astype(float)
        
        self.baseline_proportions_ = counts / counts.sum()
        self.is_fitted_ = True

        return self


    def evaluate_group(self, group_scaled: np.ndarray) -> Dict[str, float]:
        """
        Calculates group-level discrepancy metrics against the baseline distribution.

        Assigns query points to clusters to obtain observed proportions and
        evaluates Total Variation distance, Jensen-Shannon divergence, and the
        Chi-Square goodness-of-fit statistic against the expected baseline.

        Args:
            group_scaled (np.ndarray): Scaled evaluation batch of shape
                `(n_group_samples, n_features)`.

        Returns:
            Dict[str, Any]: Dictionary containing computed discrepancy metrics:
                - `'total_variation'`: Total Variation distance in [0.0, 1.0].
                - `'jensen_shannon_divergence'`: Jensen-Shannon divergence in [0.0, ln(2)].
                - `'chi_square_statistic'`: Chi-Square test statistic.
                - `'observed_proportions'`: List of empirical cluster proportions
                  observed in the group.

        Raises:
            RuntimeError: If called before fitting the baseline via `fit`.
        """
        if not self.is_fitted_ or self.kmeans_ is None or self.baseline_proportions_ is None:
            raise RuntimeError("You must call .fit() before evaluating clusters.")

        group_labels = self.kmeans_.predict(group_scaled)
        counts = np.bincount(group_labels, minlength=self.n_clusters)
        observed = counts / max(1, counts.sum())

        expected = np.clip(self.baseline_proportions_, 1e-9, None)
        expected = expected / expected.sum()

        tv = float(0.5 * np.abs(observed - expected).sum())

        midpoint = 0.5 * (observed + expected)
        observed_safe = np.clip(observed, 1e-12, None)
        midpoint_safe = np.clip(midpoint, 1e-12, None)
        
        js = float(
            0.5 * np.sum(observed_safe * np.log(observed_safe / midpoint_safe))
            + 0.5 * np.sum(expected * np.log(expected / midpoint_safe))
        )

        expected_counts = np.maximum(expected * counts.sum(), 1e-9)
        chi2 = float(np.sum((counts - expected_counts) ** 2 / expected_counts))

        return {
            "total_variation": tv,
            "jensen_shannon_divergence": js,
            "chi_square_statistic": chi2,
            "observed_proportions": observed.tolist(),
        }