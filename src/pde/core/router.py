
from typing import Any, Dict
from .profiler import MetaDataProfiler


class AutoPipelineRouter:
    """
    Routes input data to its specialized algorithmic family.

    Uses `MetaDataProfiler` to inspect raw tabular or sequential structures
    and identify the optimal processing pipeline.

    Attributes:
        last_profile_ (Dict[str, Any]): Metadata profile dictionary obtained
            from the most recent call to `route`. Empty before the first run.
    """

    def __init__(self) -> None:
        """
        Initializes an empty AutoPipelineRouter instance.
        """
        self.last_profile_: Dict[str, Any] = {}


    def route(self, data_input: Any) -> str:
        """
        Profiles the input data and returns the recommended pipeline name.

        Args:
            data_input (Any): Raw input data to profile and route (e.g., pandas
                DataFrame, dictionary, list of sequences/trajectories).

        Returns:
            str: Identifier name of the recommended downstream pipeline
                (e.g., `'Semi_Markov_Conformal_Engine'`,
                `'Heterogeneous_Gower_KMedoids_Engine'`, or
                `'Continuous_Conformal_Geometric_Engine'`).
        """
        self.last_profile_ = MetaDataProfiler.profile(data_input)

        return self.last_profile_["recommended_pipeline"]