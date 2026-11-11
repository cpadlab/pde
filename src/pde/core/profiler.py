
from typing import Any, Dict, List, Tuple
import pandas as pd


class MetaDataProfiler:
    """
    Inspects tabular or sequential datasets and extracts metadata without leakage.

    Infers data structures, separates continuous numerical variables from nominal
    categorical ones using heuristic cardinality checks, detects trajectory or
    sequential time-interval inputs, and routes workloads to the optimal engine.
    """


    @staticmethod
    def _profile_sequential(data_input: Any) -> Dict[str, Any] | None:
        """
        Profiles sequential trajectory inputs if the input format matches.

        Detects whether the input structure represents discrete event sequences
        or temporal event-duration trajectories, establishing the sample count,
        feature dimensionality, and the recommended sequential pipeline.

        Args:
            data_input (Any): Raw input object to evaluate. Expected to be a
                non-empty list of sequence records (e.g., lists or 2-tuples).

        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary containing:
                - `'data_modality'`: String identifier (`'sequential_temporal'`
                  or `'sequential_discrete'`).
                - `'n_samples'`: Number of sequential trajectory records.
                - `'n_features'`: Feature dimension (`2` if holding intervals
                  are present, `1` otherwise).
                - `'numeric_features'`: Empty list for sequential modalities.
                - `'categorical_features'`: Empty list for sequential modalities.
                - `'recommended_pipeline'`: Identifier set to
                  `'Semi_Markov_Conformal_Engine'`.
                Returns None if `data_input` does not match the sequential layout.
        """
        if not (isinstance(data_input, list) and len(data_input) > 0):
            return None

        first_item = data_input[0]
        if not isinstance(first_item, (tuple, list)):
            return None

        has_intervals = isinstance(first_item, tuple) and len(first_item) == 2
        modality = "sequential_temporal" if has_intervals else "sequential_discrete"
        n_features = 2 if has_intervals else 1

        return {
            "data_modality": modality,
            "n_samples": len(data_input),
            "n_features": n_features,
            "numeric_features": [],
            "categorical_features": [],
            "recommended_pipeline": "Semi_Markov_Conformal_Engine",
        }


    @staticmethod
    def _is_categorical_column(series: pd.Series, n_rows: int) -> bool:
        """
        Checks whether a column is categorical based on dtype and cardinality heuristics.

        Non-numeric series are identified as categorical by default. Numeric series
        are reclassified as categorical if they exhibit very low cardinality
        (5 or fewer distinct values in datasets containing more than 50 rows).

        Args:
            series (pd.Series): DataFrame column series to inspect.
            n_rows (int): Total number of rows in the parent DataFrame.

        Returns:
            bool: True if the series qualifies as categorical; False if it
                should be processed as a continuous numerical variable.
        """
        if not pd.api.types.is_numeric_dtype(series):
            return True

        return bool(series.nunique() <= 5 and n_rows > 50)


    @classmethod
    def _split_column_types(cls, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Categorizes DataFrame columns into numerical and categorical feature name lists.

        Iterates through all columns of the input DataFrame and applies
        cardinality heuristics via `_is_categorical_column` to partition
        features without causing data leakage.

        Args:
            df (pd.DataFrame): Tabular dataset whose columns are to be partitioned.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing:
                - **numeric_cols** (`List[str]`): Column names identified as
                  continuous numerical features.
                - **categorical_cols** (`List[str]`): Column names identified as
                  nominal or discrete categorical features.
        """
        n_rows = len(df)
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []

        for col in df.columns:
            col_str = str(col)
            if cls._is_categorical_column(df[col], n_rows):
                categorical_cols.append(col_str)
            else:
                numeric_cols.append(col_str)

        return numeric_cols, categorical_cols


    @classmethod
    def profile(cls, data_input: Any) -> Dict[str, Any]:
        """
        Profiles raw data inputs and recommends a downstream processing pipeline.

        Evaluates input formats to distinguish between sequential temporal trajectories,
        purely continuous tabular datasets, and mixed/heterogeneous tabular data.
        Applies a low-cardinality heuristic (<= 5 unique values when sample size > 50)
        to reclassify discrete numerical columns as categorical.

        Args:
            data_input (Any): Input dataset to inspect. Can be a pandas DataFrame,
                an array-like structure convertible to a DataFrame, or a list of
                sequential trajectory sequences/tuples.

        Returns:
            Dict[str, Any]: Dictionary containing inferred dataset metadata:
                - `'data_modality'`: Categorization string (`'sequential_temporal'`,
                  `'sequential_discrete'`, `'tabular_heterogeneous'`, or
                  `'tabular_continuous'`).
                - `'n_samples'`: Number of observations or sequence instances.
                - `'n_features'`: Total number of measured dimensions or columns.
                - `'numeric_features'`: List of column names classified as continuous.
                - `'categorical_features'`: List of column names classified as categorical.
                - `'recommended_pipeline'`: Recommended pipeline identifier string
                  (`'Semi_Markov_Conformal_Engine'`,
                  `'Heterogeneous_Gower_KMedoids_Engine'`, or
                  `'Continuous_Conformal_Geometric_Engine'`).
        """
        seq_profile = cls._profile_sequential(data_input)
        if seq_profile is not None:
            return seq_profile

        df = data_input if isinstance(data_input, pd.DataFrame) else pd.DataFrame(data_input)
        n_rows, n_cols = df.shape
        numeric_cols, categorical_cols = cls._split_column_types(df)

        if len(categorical_cols) > 0:
            modality = "tabular_heterogeneous"
            pipeline = "Heterogeneous_Gower_KMedoids_Engine"
        else:
            modality = "tabular_continuous"
            pipeline = "Continuous_Conformal_Geometric_Engine"

        return {
            "data_modality": modality,
            "n_samples": n_rows,
            "n_features": n_cols,
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "recommended_pipeline": pipeline,
        }
