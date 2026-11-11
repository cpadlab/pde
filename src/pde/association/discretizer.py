
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd


class QuantileDiscretizer:
    """
    Discretizer for numerical features fitted on the training dataset.

    Bins continuous variables into discrete intervals using empirical
    quantiles computed from training data, and transforms inputs into
    one-hot encoded boolean indicator features.

    Attributes:
        n_bins (int): Number of quantile bins to partition the feature space into.
        labels (List[str]): Ordered categorical labels assigned to the bins.
        bin_edges_ (Dict[str, np.ndarray]): Mapping from feature names to their
            calculated bin boundary arrays, bounded by [-inf, inf].
        feature_names_ (List[str]): List of column names selected and fitted
            for discretization.
        is_fitted_ (bool): Flag indicating whether the discretizer has been
            fitted with training data.
    """


    def __init__(self, n_bins: int = 3, labels: Optional[Sequence[str]] = None):
        """
        Initializes the QuantileDiscretizer instance.

        Args:
            n_bins (int, optional): Number of quantile intervals to form.
                Defaults to 3.
            labels (Optional[Sequence[str]], optional): Category names for each
                bin ordered from lowest to highest value. If None, defaults to
                `["Low", "Medium", "High"]`. Length must match `n_bins`.

        Raises:
            ValueError: If the number of provided labels does not match `n_bins`.
        """
        self.n_bins = int(n_bins)
        self.labels = list(labels) if labels else ["Low", "Medium", "High"]
        
        if len(self.labels) != self.n_bins:
            raise ValueError(f"The number of labels ({len(self.labels)}) must match n_bins ({self.n_bins}).")

        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False


    def fit(self, df: pd.DataFrame, features: Optional[Sequence[str]] = None) -> "QuantileDiscretizer":
        """
        Computes bin cutoffs using empirical quantiles strictly from training data.

        If quantile boundaries collapse due to heavily concentrated data, falls
        back to linearly spaced thresholds between the column's minimum and
        maximum values. The outer edges are extended to `-inf` and `inf` to
        accommodate out-of-distribution values at inference time.

        Args:
            df (pd.DataFrame): Training DataFrame used to compute feature
                quantiles.
            features (Optional[Sequence[str]], optional): Specific numeric column
                names to discretize. If None, automatically selects all numeric
                columns in `df`. Defaults to None.

        Returns:
            QuantileDiscretizer: The fitted discretizer instance (`self`).
        """
        self.feature_names_ = list(features) if features else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.bin_edges_ = {}

        quantiles = np.linspace(0, 1, self.n_bins + 1)

        for col in self.feature_names_:
            edges = np.quantile(df[col].to_numpy(), quantiles)
            edges = np.unique(edges)
            
            if len(edges) < self.n_bins + 1:
                edges = np.linspace(df[col].min(), df[col].max(), self.n_bins + 1)

            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_[col] = edges

        self.is_fitted_ = True

        return self


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms continuous features into one-hot encoded boolean indicators.

        Applies learned bin intervals to partition continuous variables and
        generates binary dummy variables for each bin category.

        Args:
            df (pd.DataFrame): DataFrame containing the columns to discretize.

        Returns:
            pd.DataFrame: Concatenated DataFrame composed of boolean one-hot
                indicator columns with names formatted as `{column}_{label}`.

        Raises:
            RuntimeError: If called before fitting the discretizer with `fit`.
        """
        if not self.is_fitted_:
            raise RuntimeError("You must call .fit() before transforming the data.")

        one_hot_parts = []
        for col in self.feature_names_:
            binned = pd.cut(df[col], bins=self.bin_edges_[col], labels=self.labels, include_lowest=True)
            dummies = pd.get_dummies(binned, prefix=col, dtype=bool)
            one_hot_parts.append(dummies)

        return pd.concat(one_hot_parts, axis=1)


    def to_dict(self) -> dict:
        """
        Serializes the discretizer configuration and learned bin edges.

        Returns:
            dict: Dictionary containing `n_bins`, `labels`, `feature_names`,
                `bin_edges` (with values serialized to lists of floats), and
                `is_fitted`.
        """
        return {
            "n_bins": self.n_bins,
            "labels": self.labels,
            "feature_names": self.feature_names_,
            "bin_edges": {k: v.tolist() for k, v in self.bin_edges_.items()},
            "is_fitted": self.is_fitted_,
        }


    @classmethod
    def from_dict(cls, data: dict) -> "QuantileDiscretizer":
        """
        Reconstructs a QuantileDiscretizer instance from a serialized dictionary.

        Args:
            data (dict): Dictionary with keys `'n_bins'`, `'labels'`,
                `'feature_names'`, `'bin_edges'`, and `'is_fitted'`.

        Returns:
            QuantileDiscretizer: A restored discretizer instance with loaded
                features and bin boundary thresholds.
        """
        inst = cls(n_bins=data["n_bins"], labels=data["labels"])
        inst.feature_names_ = data["feature_names"]
        inst.bin_edges_ = {k: np.array(v) for k, v in data["bin_edges"].items()}
        inst.is_fitted_ = bool(data["is_fitted"])
        
        return inst