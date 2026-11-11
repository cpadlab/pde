
import base64
import io
from typing import Any, Dict, Optional
import hdbscan
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

MODEL_HDBSCAN = "HDBSCAN"
MODEL_ISOLATION_FOREST = "Isolation Forest"
MODEL_LOF = "LOF"
MODEL_OCSVM = "One-Class SVM"


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


    def fit(self, x_train: np.ndarray) -> "GeometricModelsSuite":
        """
        Fits the scaler and all four detectors strictly on training data.

        Learns empirical centering and scaling parameters, then trains HDBSCAN,
        Isolation Forest, novelty-mode LOF, and One-Class SVM on the scaled features.

        Args:
            x_train (np.ndarray): 2D array of continuous training data of shape
                `(n_samples, n_features)`.

        Returns:
            GeometricModelsSuite: The fitted model suite instance (`self`).

        Raises:
            ValueError: If `x_train` is not a 2D array.
        """
        x_data = np.asarray(x_train, dtype=float)
        if x_data.ndim != 2:
            raise ValueError(f"x_train must be two-dimensional, with dimensions {x_data.ndim}D.")

        self.scaler_ = StandardScaler().fit(x_data)
        x_scaled = self.scaler_.transform(x_data)

        self.models_ = {
            MODEL_HDBSCAN: hdbscan.HDBSCAN(
                min_cluster_size=10, min_samples=5, metric="euclidean", prediction_data=True
            ).fit(x_scaled),
            MODEL_ISOLATION_FOREST: IsolationForest(
                n_estimators=250, contamination="auto", random_state=self.random_state
            ).fit(x_scaled),
            MODEL_LOF: LocalOutlierFactor(
                n_neighbors=20, novelty=True, metric="euclidean"
            ).fit(x_scaled),
            MODEL_OCSVM: OneClassSVM(
                kernel="rbf", gamma="scale", nu=0.05
            ).fit(x_scaled),
        }

        self.is_fitted_ = True

        return self


    def compute_raw_scores(self, x_query: np.ndarray) -> Dict[str, np.ndarray]:
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
            x_query (np.ndarray): 2D array of query samples of shape
                `(n_query_samples, n_features)` to evaluate.

        Returns:
            Dict[str, np.ndarray]: Mapping from detector name to a 1D array of
                shape `(n_query_samples,)` containing raw anomaly scores.

        Raises:
            RuntimeError: If called before fitting the suite with `fit`.
        """
        if not self.is_fitted_ or self.scaler_ is None:
            raise RuntimeError("You must execute .fit() before computing scores.")

        x_data = np.asarray(x_query, dtype=float)
        x_scaled = self.scaler_.transform(x_data)
        scores = {}

        labels, strengths = hdbscan.approximate_predict(self.models_[MODEL_HDBSCAN], x_scaled)
        hdb_scores = 1.0 - strengths
        hdb_scores[labels == -1] = 1.0

        scores[MODEL_HDBSCAN] = hdb_scores
        scores[MODEL_ISOLATION_FOREST] = -self.models_[MODEL_ISOLATION_FOREST].score_samples(x_scaled)
        scores[MODEL_LOF] = -self.models_[MODEL_LOF].score_samples(x_scaled)
        scores[MODEL_OCSVM] = -self.models_[MODEL_OCSVM].decision_function(x_scaled).ravel()

        return scores

    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the suite parameters, scaler, and fitted models into a JSON-safe dictionary.

        Returns:
            Dict[str, Any]: Serialized dictionary containing the metadata and binary payload.
        """
        buffer = io.BytesIO()
        joblib.dump({"scaler": self.scaler_, "models": self.models_}, buffer)
        payload = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "random_state": self.random_state,
            "is_fitted": self.is_fitted_,
            "payload": payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometricModelsSuite":
        """
        Reconstructs a GeometricModelsSuite instance from its serialized dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary output from `to_dict()`.

        Returns:
            GeometricModelsSuite: Fully restored suite instance with trained models.
        """
        instance = cls(random_state=data.get("random_state", 42))
        instance.is_fitted_ = data.get("is_fitted", False)

        if "payload" in data and data["payload"]:
            buffer = io.BytesIO(base64.b64decode(data["payload"].encode("utf-8")))
            artifacts = joblib.load(buffer)
            instance.scaler_ = artifacts.get("scaler")
            instance.models_ = artifacts.get("models", {})

        return instance