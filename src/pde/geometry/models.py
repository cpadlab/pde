
from typing import Dict, Optional
import hdbscan
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class GeometricModelsSuite:
    """
    Fits and scores the geometric quartet of anomaly detection algorithms.

    Applies standard feature scaling exclusively learned from training data and
    fits four complementary unsupervised detectors: density-based (HDBSCAN),
    isolation-based (Isolation Forest), local density-ratio (LOF), and boundary-based
    (One-Class SVM).

    Attributes:
        random_state (Optional[int]): Seed used for reproducible components.
        scaler_ (Optional[StandardScaler]): Fitted feature scaler instance. None
            before `fit`.
        models_ (Dict[str, object]): Mapping from detector names to fitted model
            instances. Empty until `fit` is called.
        is_fitted_ (bool): Flag indicating whether the scaler and detectors
            have been fitted.
    """


    def __init__(self, random_state: Optional[int] = 42):
        """
        Initializes the GeometricModelsSuite instance.

        Args:
            random_state (Optional[int], optional): Random seed passed to
                stochastic estimators like Isolation Forest. Defaults to 42.
        """
        self.random_state = random_state

        self.scaler_: Optional[StandardScaler] = None
        self.models_: Dict[str, object] = {}
        self.is_fitted_: bool = False


    def fit(self, X_train: np.ndarray) -> "GeometricModelsSuite":
        """
        Fits the scaler and all four detectors strictly on training data.

        Learns empirical centering and scaling parameters, then trains HDBSCAN,
        Isolation Forest, novelty-mode LOF, and One-Class SVM on the scaled features.

        Args:
            X_train (np.ndarray): 2D array of continuous training data of shape
                `(n_samples, n_features)`.

        Returns:
            GeometricModelsSuite: The fitted model suite instance (`self`).

        Raises:
            ValueError: If `X_train` is not a 2D array.
        """
        X = np.asarray(X_train, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X_train must be two-dimensional, with dimensions {X.ndim}D.")

        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)

        self.models_ = {
            "HDBSCAN": hdbscan.HDBSCAN(
                min_cluster_size=10, min_samples=5, metric="euclidean", prediction_data=True
            ).fit(X_scaled),
            "Isolation Forest": IsolationForest(
                n_estimators=250, contamination="auto", random_state=self.random_state
            ).fit(X_scaled),
            "LOF": LocalOutlierFactor(
                n_neighbors=20, novelty=True, metric="euclidean"
            ).fit(X_scaled),
            "One-Class SVM": OneClassSVM(
                kernel="rbf", gamma="scale", nu=0.05
            ).fit(X_scaled)
        }

        self.is_fitted_ = True

        return self


    def compute_raw_scores(self, X_query: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes raw non-conformity anomaly scores for each detector.

        Transforms query observations using the learned training scaler and
        evaluates algorithm-specific non-conformity scores oriented such that
        higher values denote higher spatial abnormality or rarity.

        Scoring logic per algorithm:
            - **HDBSCAN**: `1.0 - membership_strength` (unclustered noise set to 1.0).
            - **Isolation Forest**: `-score_samples(X)`.
            - **LOF**: `-score_samples(X)`.
            - **One-Class SVM**: `-decision_function(X)`.

        Args:
            X_query (np.ndarray): 2D array of query samples of shape
                `(n_query_samples, n_features)` to evaluate.

        Returns:
            Dict[str, np.ndarray]: Mapping from detector name to a 1D array of
                shape `(n_query_samples,)` containing raw anomaly scores.

        Raises:
            RuntimeError: If called before fitting the suite with `fit`.
        """
        if not self.is_fitted_ or self.scaler_ is None:
            raise RuntimeError("Debe ejecutar .fit() antes de calcular scores.")

        X = np.asarray(X_query, dtype=float)
        X_scaled = self.scaler_.transform(X)
        scores = {}

        labels, strengths = hdbscan.approximate_predict(self.models_["HDBSCAN"], X_scaled)
        hdb_scores = 1.0 - strengths
        hdb_scores[labels == -1] = 1.0

        scores["HDBSCAN"] = hdb_scores
        scores["Isolation Forest"] = -self.models_["Isolation Forest"].score_samples(X_scaled)
        scores["LOF"] = -self.models_["LOF"].score_samples(X_scaled)
        scores["One-Class SVM"] = -self.models_["One-Class SVM"].decision_function(X_scaled).ravel()

        return scores