
from typing import Any, Dict, List
import pandas as pd


class MetaDataProfiler:
    """
    Inspects tabular or sequential datasets and extracts metadata without leakage.

    Infers data structures, separates continuous numerical variables from nominal
    categorical ones using heuristic cardinality checks, detects trajectory or
    sequential time-interval inputs, and routes workloads to the optimal engine.
    """


    @staticmethod
    def profile(data_input: Any) -> Dict[str, Any]:
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
        if isinstance(data_input, list) and len(data_input) > 0:
            first_item = data_input[0]

            if isinstance(first_item, (tuple, list)):
                has_intervals = isinstance(first_item, tuple) and len(first_item) == 2

                return {
                    "data_modality": "sequential_temporal" if has_intervals else "sequential_discrete",
                    "n_samples": len(data_input),
                    "n_features": 2 if has_intervals else 1,
                    "numeric_features": [],
                    "categorical_features": [],
                    "recommended_pipeline": "Semi_Markov_Conformal_Engine",
                }

        df = data_input if isinstance(data_input, pd.DataFrame) else pd.DataFrame(data_input)
        n_rows, n_cols = df.shape

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []

        for col in df.columns:
            col_str = str(col)
            if pd.api.types.is_numeric_dtype(df[col]):
                
                n_unique = df[col].nunique()

                if n_unique <= 5 and n_rows > 50:
                    categorical_cols.append(col_str)
                else:
                    numeric_cols.append(col_str)
            else:
                categorical_cols.append(col_str)

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