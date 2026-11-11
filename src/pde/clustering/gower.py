
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd


class GowerDistanceMetric:
    """
    Computes Gower distance matrices normalized by training feature ranges.

    Calculates pairwise distances for mixed-type tabular datasets by computing
    range-normalized absolute differences for continuous features and 0/1 matching
    distances for nominal categorical features.

    Attributes:
        numeric_features (List[str]): List of column names treated as numerical.
        categorical_features (List[str]): List of column names treated as
            categorical.
        ranges_ (Dict[str, float]): Feature ranges (max - min) learned during `fit`
            for each numerical column, bounded below by 1e-9.
        mins_ (Dict[str, float]): Minimum values learned during `fit` for each
            numerical column.
        is_fitted_ (bool): Flag indicating whether the metric ranges have been
            fitted with training data.
    """


    def __init__(
        self, numeric_features: Optional[Sequence[str]] = None, 
        categorical_features: Optional[Sequence[str]] = None
    ):
        """
        Initializes the GowerDistanceMetric instance.

        Args:
            numeric_features (Optional[Sequence[str]], optional): Names of numeric
                columns to include in the distance computation. Defaults to None.
            categorical_features (Optional[Sequence[str]], optional): Names of
                categorical columns to include in the distance computation.
                Defaults to None.
        """
        self.numeric_features: List[str] = list(numeric_features) if numeric_features else []
        self.categorical_features: List[str] = list(categorical_features) if categorical_features else []
        self.ranges_: Dict[str, float] = {}
        self.mins_: Dict[str, float] = {}
        self.is_fitted_: bool = False


    def fit(self, df: pd.DataFrame) -> "GowerDistanceMetric":
        """
        Learns feature types and numeric ranges exclusively from training data.

        If no feature names were specified during initialization, automatically
        infers numeric and categorical columns from the DataFrame dtypes.

        Args:
            df (pd.DataFrame): Training DataFrame used to infer feature categories
                and establish numerical normalization ranges.

        Returns:
            GowerDistanceMetric: The fitted metric instance (`self`).
        """
        if not self.numeric_features and not self.categorical_features:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    self.numeric_features.append(str(col))
                else:
                    self.categorical_features.append(str(col))

        self.ranges_ = {}
        self.mins_ = {}
        for col in self.numeric_features:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_range = max(1e-9, col_max - col_min)
            self.ranges_[col] = col_range
            self.mins_[col] = col_min

        self.is_fitted_ = True
        
        return self


    def compute_distance_matrix(self, df_a: pd.DataFrame, df_b: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Computes the pairwise Gower distance matrix between two datasets.

        Evaluates normalized Manhattan distance for continuous variables and simple
        matching dissimilarity for categorical variables, averaging across all
        defined features.

        Args:
            df_a (pd.DataFrame): Primary DataFrame of shape `(N, D)`.
            df_b (Optional[pd.DataFrame], optional): Secondary DataFrame of shape
                `(M, D)`. If None, computes the intra-dataset pairwise distance
                matrix of shape `(N, N)` for `df_a`. Defaults to None.

        Returns:
            np.ndarray: 2D array of shape `(N, M)` containing normalized Gower
                distances bounded within [0.0, 1.0].

        Raises:
            RuntimeError: If called before fitting the metric with `fit`.
            ValueError: If no numerical or categorical features are defined.
        """
        if not self.is_fitted_:
            raise RuntimeError("You must call .fit() before calculating distance matrices.")

        target_b = df_a if df_b is None else df_b
        n_features = len(self.numeric_features) + len(self.categorical_features)
        if n_features == 0:
            raise ValueError("There are no defined characteristics for the Gower calculation.")

        N = len(df_a)
        M = len(target_b)
        distance_accum = np.zeros((N, M), dtype=float)

        for col in self.numeric_features:
            vals_a = df_a[col].to_numpy()[:, np.newaxis]
            vals_b = target_b[col].to_numpy()[np.newaxis, :]
            r = self.ranges_[col]
            diff = np.abs(vals_a - vals_b) / r
            distance_accum += np.clip(diff, 0.0, None)

        for col in self.categorical_features:
            cat_a = df_a[col].to_numpy()[:, np.newaxis]
            cat_b = target_b[col].to_numpy()[np.newaxis, :]
            distance_accum += (cat_a != cat_b).astype(float)

        return distance_accum / float(n_features)


    def to_dict(self) -> dict:
        """
        Serializes the metric configuration and fitted ranges into a dictionary.

        Returns:
            dict: Dictionary containing feature names, normalization ranges,
                minimum values, and calibration status.
        """
        return {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "ranges": self.ranges_,
            "mins": self.mins_,
            "is_fitted": self.is_fitted_,
        }


    @classmethod
    def from_dict(cls, data: dict) -> "GowerDistanceMetric":
        """
        Reconstructs a GowerDistanceMetric instance from a serialized dictionary.

        Args:
            data (dict): Dictionary with keys `'numeric_features'`,
                `'categorical_features'`, `'ranges'`, `'mins'`, and `'is_fitted'`.

        Returns:
            GowerDistanceMetric: A restored instance initialized with the saved
                attributes and learned ranges.
        """
        inst = cls(data["numeric_features"], data["categorical_features"])
        inst.ranges_ = {k: float(v) for k, v in data["ranges"].items()}
        inst.mins_ = {k: float(v) for k, v in data["mins"].items()}
        inst.is_fitted_ = bool(data["is_fitted"])

        return inst