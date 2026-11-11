
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest

import pde.persistence
from pde.persistence import ModelSerializer


@pytest.mark.unit
class TestPersistenceModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.persistence.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains ModelSerializer.
        """
        expected = ["ModelSerializer"]
        assert sorted(pde.persistence.__all__) == sorted(expected)
        assert pde.persistence.ModelSerializer is ModelSerializer


@pytest.mark.unit
class TestModelSerializer:
    """
    Unit test suite for ModelSerializer recursive serialization, disk I/O, and deserialization.
    """

    def test_convert_to_serializable_numpy_primitives(self) -> None:
        """
        Verifies recursive conversion of NumPy scalar integers, floats, and ndarrays.
        """
        np_int = np.int64(42)
        conv_int = ModelSerializer._convert_to_serializable(np_int)
        assert isinstance(conv_int, int)
        assert conv_int == 42

        np_float = np.float64(3.14159)
        conv_float = ModelSerializer._convert_to_serializable(np_float)
        assert isinstance(conv_float, float)
        assert np.isclose(conv_float, 3.14159)

        np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        conv_arr = ModelSerializer._convert_to_serializable(np_arr)
        assert isinstance(conv_arr, list)
        assert conv_arr == [[1.0, 2.0], [3.0, 4.0]]

    def test_convert_to_serializable_nested_structures(self) -> None:
        """
        Verifies deep conversion of nested dictionaries, lists, and tuples containing NumPy types.
        """
        nested_data = {
            1: np.array([10, 20, 30], dtype=np.int32),
            "scalars": (np.float32(1.5), np.int64(99)),
            "deep": {
                "vector": np.array([0.1, 0.2]),
                "plain_str": "constant",
                "plain_bool": True,
                "none_val": None,
            },
        }

        converted = ModelSerializer._convert_to_serializable(nested_data)

        assert isinstance(converted, dict)
        assert "1" in converted
        assert converted["1"] == [10, 20, 30]

        assert isinstance(converted["scalars"], list)
        assert converted["scalars"] == [1.5, 99]
        assert isinstance(converted["scalars"][0], float)
        assert isinstance(converted["scalars"][1], int)

        assert converted["deep"]["vector"] == [0.1, 0.2]
        assert converted["deep"]["plain_str"] == "constant"
        assert converted["deep"]["plain_bool"] is True
        assert converted["deep"]["none_val"] is None

    def test_convert_to_serializable_passthrough_primitives(self) -> None:
        """
        Verifies that standard Python primitives (strings, booleans, None) pass through unchanged.
        """
        assert ModelSerializer._convert_to_serializable("hello") == "hello"
        assert ModelSerializer._convert_to_serializable(True) is True
        assert ModelSerializer._convert_to_serializable(None) is None
        assert ModelSerializer._convert_to_serializable(123) == 123
        assert ModelSerializer._convert_to_serializable(4.56) == 4.56

    def test_save_json_creates_directories_and_writes_valid_json(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Verifies that save_json creates nested parent directories and writes formatted JSON.
        """
        target_file = tmp_path / "deep" / "nested" / "dir" / "model_state.json"
        assert not target_file.parent.exists()

        data = {
            "model_name": "TestEngine",
            "weights": np.array([0.25, 0.75]),
            "iterations": np.int64(100),
        }

        saved_path = ModelSerializer.save_json(data, target_file)

        assert saved_path == target_file
        assert target_file.exists()
        assert target_file.is_file()

        content = target_file.read_text(encoding="utf-8")
        assert '"model_name": "TestEngine"' in content
        assert '"weights": [' in content
        assert '"iterations": 100' in content

    def test_load_json_raises_file_not_found_error_on_missing_path(
        self,
        tmp_path: Path,
    ) -> None:
        """
        Ensures load_json raises FileNotFoundError when the requested artifact does not exist.
        """
        non_existent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            ModelSerializer.load_json(non_existent)

    def test_save_and_load_json_roundtrip_with_fixture(
        self,
        tmp_model_path: Path,
    ) -> None:
        """
        Verifies end-to-end save and load roundtrip fidelity using the conftest fixture.
        """
        original_state: Dict[str, Any] = {
            "engine": "Semi_Markov_Conformal_Engine",
            "alpha": 0.05,
            "transition_matrix": np.array(
                [
                    [0.1, 0.9],
                    [0.8, 0.2],
                ]
            ),
            "duration_parameters": {
                "0": (np.float64(1.5), np.float64(0.3)),
                "1": (np.float64(2.5), np.float64(0.4)),
            },
            "calibrated": True,
        }

        ModelSerializer.save_json(original_state, tmp_model_path)
        loaded_state = ModelSerializer.load_json(tmp_model_path)

        assert loaded_state["engine"] == original_state["engine"]
        assert loaded_state["alpha"] == original_state["alpha"]
        assert loaded_state["calibrated"] is True
        assert loaded_state["transition_matrix"] == [[0.1, 0.9], [0.8, 0.2]]
        assert loaded_state["duration_parameters"]["0"] == [1.5, 0.3]
        assert loaded_state["duration_parameters"]["1"] == [2.5, 0.4]
