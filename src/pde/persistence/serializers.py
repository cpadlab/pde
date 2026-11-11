
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np


class ModelSerializer:
    """
    Universal serializer for JSON-compatible model state dictionaries.

    Provides utilities to recursively convert NumPy arrays and scalar types
    into standard Python serializable objects, as well as saving and loading
    serialized state dictionaries to and from disk.
    """


    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """
        Recursively converts NumPy types and nested structures into serializable objects.

        Args:
            obj (Any): Object or nested structure containing potential NumPy
                arrays or scalar values to convert.

        Returns:
            Any: JSON-serializable version of the input object with NumPy types
                mapped to standard Python primitives (e.g., lists, ints, floats).
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)

        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)

        if isinstance(obj, dict):
            return {str(k): ModelSerializer._convert_to_serializable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [ModelSerializer._convert_to_serializable(item) for item in obj]

        return obj


    @classmethod
    def save_json(cls, state_dict: Dict[str, Any], filepath: Path) -> Path:
        """
        Saves a model state dictionary to a JSON file.

        Creates necessary parent directories if they do not exist and serializes
        the state dictionary with 4-space indentation.

        Args:
            state_dict (Dict[str, Any]): Model state dictionary containing parameters,
                configurations, and weights.
            filepath (Path): Target file path where the JSON file will be written.

        Returns:
            Path: The resolved file path where the artifact was stored.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_data = cls._convert_to_serializable(state_dict)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=4)

        return filepath


    @staticmethod
    def load_json(filepath: Path) -> Dict[str, Any]:
        """
        Loads a model state dictionary from a JSON file.

        Args:
            filepath (Path): Path to the serialized JSON model artifact.

        Returns:
            Dict[str, Any]: Deserialized model state dictionary.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"The template file does not exist: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)