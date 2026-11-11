
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

from pde.core.profiler import MetaDataProfiler
from pde.calibration.conformal import ConformalCalibrator
from pde.calibration.selective import SelectiveClassifier
from pde.drift.monitor import DistributionDriftMonitor
from pde.geometry.models import GeometricModelsSuite
from pde.clustering.gower import GowerDistanceMetric
from pde.sequential.markov import SemiMarkovEngine
from pde.persistence.serializers import ModelSerializer


class PatternDiscoveryEngine:
    """
    Production API for the Pattern Discovery Engine (PDE).

    Enables inductive fitting, selective conformal inference, runtime
    distributional drift monitoring, and autonomous re-adaptation across
    continuous tabular, heterogeneous mixed-type, and sequential/temporal
    modalities.

    Attributes:
        VERSION (str): Engine software release version identifier.
        alpha (float): Nominal significance level for anomaly confirmation.
        review_boundary (float): Upper p-value threshold for the manual review
            region.
        auto_adapt (bool): Flag determining whether the engine automatically
            refits upon detecting significant drift.
        random_state (int): Random seed passed to internal stochastic
            estimators.
        calibrator (ConformalCalibrator): Inductive conformal calibrator
            instance.
        classifier (SelectiveClassifier): Three-way selective classifier
            instance.
        drift_monitor (DistributionDriftMonitor): Kolmogorov-Smirnov drift
            monitor tracking p-value uniformity against U(0, 1).
        is_fitted (bool): Flag indicating whether the engine has been trained
            and calibrated.
        modality (Optional[str]): Inferred data modality string.
        pipeline_name (Optional[str]): Selected algorithmic pipeline name.
        feature_names (List[str]): List of active feature names.
        numeric_features (List[str]): List of continuous numeric column names.
        categorical_features (List[str]): List of categorical column names.
        continuous_suite (Optional[GeometricModelsSuite]): Fitted continuous
            geometric quartet suite.
        gower_metric (Optional[GowerDistanceMetric]): Fitted Gower distance
            metric for mixed data.
        train_heterogeneous_ref_ (Optional[pd.DataFrame]): Reference baseline
            records retained for heterogeneous k-NN scoring.
        semi_markov_suite (Optional[SemiMarkovEngine]): Fitted Semi-Markov
            engine for sequential trajectories.
    """

    VERSION = "1.0.0"

    def __init__(
        self, alpha: float = 0.05, review_boundary: float = 0.20, 
        auto_adapt: bool = True, random_state: int = 42
    ):
        """
        Initializes the PatternDiscoveryEngine with inference hyperparameters.

        Args:
            alpha (float, optional): Anomaly significance threshold. Defaults
                to 0.05.
            review_boundary (float, optional): Ambiguity boundary for routing
                observations to manual review. Defaults to 0.20.
            auto_adapt (bool, optional): Whether to trigger autonomous refitting
                when drift is flagged. Defaults to True.
            random_state (int, optional): Random seed for reproducible
                sub-models. Defaults to 42.
        """
        self.alpha = float(alpha)
        self.review_boundary = float(review_boundary)
        self.auto_adapt = bool(auto_adapt)
        self.random_state = int(random_state)

        self.calibrator = ConformalCalibrator()
        self.classifier = SelectiveClassifier(alpha=self.alpha, review_boundary=self.review_boundary)
        self.drift_monitor = DistributionDriftMonitor(alpha_drift=0.01)

        self.is_fitted = False
        self.modality: Optional[str] = None
        self.pipeline_name: Optional[str] = None
        self.feature_names: list = []
        self.numeric_features: list = []
        self.categorical_features: list = []

        self.continuous_suite: Optional[GeometricModelsSuite] = None
        self.gower_metric: Optional[GowerDistanceMetric] = None
        self.train_heterogeneous_ref_: Optional[pd.DataFrame] = None
        self.semi_markov_suite: Optional[SemiMarkovEngine] = None


    def fit(self, train_data: Any, cal_data: Optional[Any] = None) -> "PatternDiscoveryEngine":
        """
        Profiles the input space, fits estimators, and calibrates conformal p-values.

        Inspects the structure of `train_data`, assigns the appropriate mathematical
        modality, fits the respective model family, and calibrates non-parametric
        empirical quantiles on `cal_data` (or `train_data` if omitted).

        Args:
            train_data (Any): Training dataset (e.g., pandas DataFrame, array-like,
                or list of trajectory tuples).
            cal_data (Optional[Any], optional): Independent calibration dataset
                without data leakage. If None, falls back to using `train_data`.
                Defaults to None.

        Returns:
            PatternDiscoveryEngine: The fitted and calibrated engine instance (`self`).
        """
        profile = MetaDataProfiler.profile(train_data)
        self.modality = profile["data_modality"]
        self.pipeline_name = profile["recommended_pipeline"]
        self.numeric_features = profile.get("numeric_features", [])
        self.categorical_features = profile.get("categorical_features", [])

        if self.modality == "tabular_continuous":
            df_train = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
            self.feature_names = df_train.columns.tolist()
            X_train = df_train[self.feature_names].to_numpy(dtype=float)

            self.continuous_suite = GeometricModelsSuite(random_state=self.random_state)
            self.continuous_suite.fit(X_train)

            if cal_data is not None:
                df_cal = cal_data if isinstance(cal_data, pd.DataFrame) else pd.DataFrame(cal_data)
                X_cal = df_cal[self.feature_names].to_numpy(dtype=float)
            else:
                X_cal = X_train

            raw_scores = self.continuous_suite.compute_raw_scores(X_cal)
            self.calibrator.fit(raw_scores["Isolation Forest"])

        elif self.modality == "tabular_heterogeneous":
            df_train = train_data if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)
            self.feature_names = self.numeric_features + self.categorical_features
            self.train_heterogeneous_ref_ = df_train[self.feature_names].copy().reset_index(drop=True)

            self.gower_metric = GowerDistanceMetric(
                numeric_features=self.numeric_features,
                categorical_features=self.categorical_features,
            )
            self.gower_metric.fit(self.train_heterogeneous_ref_)

            df_cal = cal_data if cal_data is not None else df_train
            cal_scores = self._score_heterogeneous(df_cal)
            self.calibrator.fit(cal_scores)

        elif "sequential" in self.modality:
            self.semi_markov_suite = SemiMarkovEngine(n_states=5, laplace_alpha=0.01)
            self.semi_markov_suite.fit(train_data)

            cal_seqs = cal_data if cal_data is not None else train_data
            t_scores, d_scores = self.semi_markov_suite.score(cal_seqs)
            cal_scores = t_scores + d_scores
            self.calibrator.fit(cal_scores)

        self.is_fitted = True

        return self


    def _score_continuous(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes raw Isolation Forest anomaly scores for continuous tabular data.

        Args:
            df (pd.DataFrame): Input DataFrame containing continuous variables.

        Returns:
            np.ndarray: 1D array of continuous non-conformity scores.
        """
        X = df[self.feature_names].to_numpy(dtype=float)
        scores = self.continuous_suite.compute_raw_scores(X)

        return scores["Isolation Forest"]


    def _score_heterogeneous(self, df: pd.DataFrame, k: int = 5) -> np.ndarray:
        """
        Computes mean k-NN Gower distances against the reference training set.

        Args:
            df (pd.DataFrame): Mixed-type query DataFrame.
            k (int, optional): Number of nearest neighbors to average across.
                Defaults to 5.

        Returns:
            np.ndarray: 1D array of mean nearest-neighbor Gower distance scores.
        """
        D = self.gower_metric.compute_distance_matrix(df, self.train_heterogeneous_ref_)

        return np.sort(D, axis=1)[:, :k].mean(axis=1)

    def _score_sequential(self, sequences: list) -> np.ndarray:
        """
        Computes combined transition and duration NLL scores for sequences.

        Args:
            sequences (list): List of sequence tuples `(events, durations)`.

        Returns:
            np.ndarray: 1D array of combined Semi-Markov non-conformity scores.
        """
        t_s, d_s = self.semi_markov_suite.score(sequences)
        
        return t_s + d_s


    def predict(self, query_data: Any) -> Dict[str, Any]:
        """
        Executes out-of-sample selective conformal inference on query observations.

        Evaluates continuous non-conformity scores, derives exact non-parametric
        p-values, and maps them to operational labels (`'anomaly'`, `'review'`,
        or `'no_alert'`).

        Args:
            query_data (Any): Query data matching the fitted modality (e.g.,
                DataFrame, array-like, or trajectory list).

        Returns:
            Dict[str, Any]: Dictionary containing inference deliverables:
                - `'scores'` (`np.ndarray`): Raw non-conformity scores.
                - `'p_values'` (`np.ndarray`): Exact conformal p-values.
                - `'decisions'` (`np.ndarray`): String array with operational
                  labels (`'anomaly'`, `'review'`, or `'no_alert'`).
                - `'is_anomaly'` (`np.ndarray`): Binary 1D integer array (`1`
                  if `'anomaly'`, `0` otherwise).

        Raises:
            RuntimeError: If called before fitting the engine via `fit`.
        """
        if not self.is_fitted:
            raise RuntimeError("Debe ejecutar .fit() antes de predecir.")

        if self.modality == "tabular_continuous":
            df = query_data if isinstance(query_data, pd.DataFrame) else pd.DataFrame(query_data)
            scores = self._score_continuous(df)
        elif self.modality == "tabular_heterogeneous":
            df = query_data if isinstance(query_data, pd.DataFrame) else pd.DataFrame(query_data)
            scores = self._score_heterogeneous(df)
        else:
            scores = self._score_sequential(query_data)

        p_values = self.calibrator.compute_pvalue(scores)
        decisions = self.classifier.decide(p_values)

        return {
            "scores": scores,
            "p_values": p_values,
            "decisions": decisions,
            "is_anomaly": (decisions == "anomaly").astype(int),
        }


    def monitor_drift(self, window_data: Any) -> Dict[str, Any]:
        """
        Evaluates p-value uniformity over a window and optionally adapts.

        Applies the Kolmogorov-Smirnov test to verify whether observed p-values
        remain uniformly distributed. If significant drift is flagged and
        `auto_adapt` is enabled, refits and recalibrates the engine on the window.

        Args:
            window_data (Any): Recent inference batch observations.

        Returns:
            Dict[str, Any]: Dictionary containing drift diagnostic metrics:
                - `'ks_statistic'` (`float`): Kolmogorov-Smirnov test statistic.
                - `'ks_pvalue'` (`float`): Goodness-of-fit p-value against U(0, 1).
                - `'drift_detected'` (`bool`): True if significant drift was flagged.
                - `'adapted'` (`bool`): True if autonomous refitting was executed.
                - `'observed_alert_rate'` (`float`): Proportion of samples with
                  `p <= alpha`.
        """
        preds = self.predict(window_data)
        eval_result = self.drift_monitor.evaluate_window(preds["p_values"], self.alpha)
        drift_flag = bool(eval_result["drift_detected"])

        adapted = False
        if drift_flag and self.auto_adapt:
            self.fit(window_data, cal_data=window_data)
            adapted = True

        return {
            "ks_statistic": eval_result["ks_statistic"],
            "ks_pvalue": eval_result["ks_pvalue"],
            "drift_detected": drift_flag,
            "adapted": adapted,
            "observed_alert_rate": eval_result["observed_alert_rate"],
        }


    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Serializes the engine state to disk as a JSON artifact.

        Encodes configurations, metadata, feature mappings, calibrator quantiles,
        and internal sub-model state dictionaries into an indented JSON file.

        Args:
            filepath (Union[str, Path]): Destination file path.

        Returns:
            Path: The resolved file path where the JSON artifact was saved.

        Raises:
            RuntimeError: If invoked on an unfitted engine.
        """
        if not self.is_fitted:
            raise RuntimeError("You cannot save an untrained model.")

        state = {
            "version": self.VERSION,
            "alpha": self.alpha,
            "review_boundary": self.review_boundary,
            "auto_adapt": self.auto_adapt,
            "random_state": self.random_state,
            "modality": self.modality,
            "pipeline_name": self.pipeline_name,
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "calibrator": self.calibrator.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.modality == "tabular_continuous" and self.continuous_suite:
            state["continuous_suite"] = self.continuous_suite.to_dict()

        elif self.modality == "tabular_heterogeneous" and self.gower_metric:
            state["gower_metric"] = self.gower_metric.to_dict()
            state["train_heterogeneous_ref"] = self.train_heterogeneous_ref_.to_dict(orient="records")

        elif self.modality and "sequential" in self.modality and self.semi_markov_suite:
            state["semi_markov"] = self.semi_markov_suite.to_dict()

        return ModelSerializer.save_json(state, Path(filepath))


    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PatternDiscoveryEngine":
        """
        Reconstructs a PatternDiscoveryEngine instance from a serialized JSON file.

        Args:
            filepath (Union[str, Path]): Path to the JSON model artifact.

        Returns:
            PatternDiscoveryEngine: Fully restored and calibrated engine instance.
        """
        state = ModelSerializer.load_json(Path(filepath))
        engine = cls(
            alpha=state["alpha"],
            review_boundary=state["review_boundary"],
            auto_adapt=state["auto_adapt"],
            random_state=state.get("random_state", 42),
        )

        engine.modality = state["modality"]
        engine.pipeline_name = state["pipeline_name"]
        engine.feature_names = state["feature_names"]
        engine.numeric_features = state["numeric_features"]
        engine.categorical_features = state["categorical_features"]
        engine.calibrator = ConformalCalibrator.from_dict(state["calibrator"])

        if engine.modality == "tabular_continuous" and "continuous_suite" in state:
            engine.continuous_suite = GeometricModelsSuite.from_dict(state["continuous_suite"])

        elif engine.modality == "tabular_heterogeneous":
            engine.gower_metric = GowerDistanceMetric.from_dict(state["gower_metric"])
            engine.train_heterogeneous_ref_ = pd.DataFrame(state["train_heterogeneous_ref"])

        elif engine.modality and "sequential" in engine.modality:
            engine.semi_markov_suite = SemiMarkovEngine.from_dict(state["semi_markov"])

        engine.is_fitted = True

        return engine