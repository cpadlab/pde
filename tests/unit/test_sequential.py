
from typing import Dict, List, Tuple
import numpy as np
import pytest

import pde.sequential
from pde.sequential import SemiMarkovEngine


@pytest.mark.unit
class TestSequentialModuleExports:
    """
    Validates package-level exports and __all__ definitions in pde.sequential.
    """

    def test_all_exports(self) -> None:
        """
        Ensures __all__ contains SemiMarkovEngine.
        """
        expected = ["SemiMarkovEngine"]
        assert sorted(pde.sequential.__all__) == sorted(expected)
        assert pde.sequential.SemiMarkovEngine is SemiMarkovEngine


@pytest.mark.unit
class TestSemiMarkovEngine:
    """
    Unit test suite for SemiMarkovEngine transition estimation, Log-Normal durations, and scoring.
    """

    def test_init_parameters(self) -> None:
        """
        Verifies default parameter initialization.
        """
        engine = SemiMarkovEngine(n_states=4, laplace_alpha=0.05)
        assert engine.n_states == 4
        assert engine.laplace_alpha == 0.05
        assert engine.p_hat_ is None
        assert engine.duration_params_ == {}
        assert not engine.is_fitted_

    def test_fit_transition_matrix_stochastic_property(self) -> None:
        """
        Verifies that estimated transition matrix p_hat_ rows sum to 1.0 (stochastic matrix).
        """
        train_seqs = [
            ([0, 1, 2], [1.0, 2.0]),
            ([1, 2, 0], [2.5, 0.5]),
            ([2, 0, 1], [3.0, 1.5]),
        ]
        engine = SemiMarkovEngine(n_states=3, laplace_alpha=0.01)
        engine.fit(train_seqs)

        assert engine.is_fitted_
        assert engine.p_hat_ is not None
        assert engine.p_hat_.shape == (3, 3)

        row_sums = engine.p_hat_.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
        assert np.all(engine.p_hat_ > 0.0)

    def test_fit_unvisited_state_duration_fallback(self) -> None:
        """
        Ensures that states with fewer than 2 observed durations fall back to default (2.0, 1.0).
        """
        train_seqs = [
            ([0, 1], [5.0]),
            ([0, 1], [6.0]),
        ]
        engine = SemiMarkovEngine(n_states=3, laplace_alpha=0.01)
        engine.fit(train_seqs)

        mu_0, sigma_0 = engine.duration_params_[0]
        assert np.isclose(mu_0, np.mean(np.log([5.0, 6.0])))

        assert engine.duration_params_[2] == (2.0, 1.0)

    def test_score_raises_runtime_error_if_not_fitted(self) -> None:
        """
        Ensures score raises RuntimeError if called prior to fit.
        """
        engine = SemiMarkovEngine()
        query = [([0, 1], [1.0])]
        with pytest.raises(RuntimeError, match="must be fitted using .fit\\(\\)"):
            engine.score(query)

    def test_score_handles_empty_duration_sequence(self) -> None:
        """
        Verifies that sequences with empty durations return (0.0, 0.0) NLL scores.
        """
        train_seqs = [
            ([0, 1], [1.0]),
            ([0, 1], [2.0]),
        ]
        engine = SemiMarkovEngine(n_states=2).fit(train_seqs)

        empty_seq = [([0], [])]
        trans_nll, dur_nll = engine.score(empty_seq)

        assert len(trans_nll) == 1
        assert len(dur_nll) == 1
        assert trans_nll[0] == 0.0
        assert dur_nll[0] == 0.0

    def test_score_detects_topological_and_temporal_anomalies(self) -> None:
        """
        Verifies that illegal jumps produce elevated transition NLL,
        and abnormal linger times produce elevated duration NLL.
        """
        train_seqs = [
            ([0, 1, 2], [2.0, 2.1]) for _ in range(50)
        ]
        engine = SemiMarkovEngine(n_states=3, laplace_alpha=0.001).fit(train_seqs)

        normal_seq = [([0, 1, 2], [2.0, 2.0])]
        norm_t_nll, norm_d_nll = engine.score(normal_seq)

        jump_seq = [([0, 2], [2.0])]
        jump_t_nll, jump_d_nll = engine.score(jump_seq)

        assert jump_t_nll[0] > norm_t_nll[0]

        stalled_seq = [([0, 1], [500.0])]
        stalled_t_nll, stalled_d_nll = engine.score(stalled_seq)

        assert stalled_d_nll[0] > norm_d_nll[0]

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """
        Verifies serialization and reconstruction fidelity of SemiMarkovEngine.
        """
        train_seqs = [
            ([0, 1, 2], [1.0, 3.0]),
            ([1, 2, 0], [2.0, 4.0]),
        ]
        engine = SemiMarkovEngine(n_states=3, laplace_alpha=0.02).fit(train_seqs)

        serialized = engine.to_dict()
        assert serialized["n_states"] == 3
        assert serialized["laplace_alpha"] == 0.02
        assert serialized["is_fitted"] is True
        assert isinstance(serialized["p_hat"], list)
        assert "0" in serialized["duration_params"]

        reconstructed = SemiMarkovEngine.from_dict(serialized)
        assert reconstructed.is_fitted_ is True
        assert reconstructed.n_states == engine.n_states
        assert reconstructed.laplace_alpha == engine.laplace_alpha
        assert np.allclose(reconstructed.p_hat_, engine.p_hat_)
        assert reconstructed.duration_params_ == engine.duration_params_

        query = [([0, 1], [1.5])]
        t_orig, d_orig = engine.score(query)
        t_recon, d_recon = reconstructed.score(query)
        assert np.allclose(t_orig, t_recon)
        assert np.allclose(d_orig, d_recon)

    def test_to_dict_unfitted_state(self) -> None:
        """
        Verifies serialization of an unfitted engine where p_hat_ is None.
        """
        engine = SemiMarkovEngine(n_states=4)
        serialized = engine.to_dict()
        assert serialized["p_hat"] is None
        assert serialized["is_fitted"] is False

        reconstructed = SemiMarkovEngine.from_dict(serialized)
        assert reconstructed.p_hat_ is None
        assert reconstructed.is_fitted_ is False

    def test_sequential_with_conftest_fixture(
        self,
        sequential_temporal_data: Tuple[
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float]]],
            List[Tuple[List[int], List[float], str]],
        ],
    ) -> None:
        """
        Verifies end-to-end fitting and scoring using the shared conftest fixture.
        """
        train_seqs, test_normal, anomalies = sequential_temporal_data
        engine = SemiMarkovEngine(n_states=4, laplace_alpha=0.01).fit(train_seqs)

        norm_t_nll, norm_d_nll = engine.score(test_normal)
        anom_seqs = [(item[0], item[1]) for item in anomalies]
        anom_t_nll, anom_d_nll = engine.score(anom_seqs)

        assert len(norm_t_nll) == len(test_normal)
        assert len(anom_t_nll) == len(anomalies)

        assert (anom_t_nll.mean() > norm_t_nll.mean()) or (anom_d_nll.mean() > norm_d_nll.mean())
