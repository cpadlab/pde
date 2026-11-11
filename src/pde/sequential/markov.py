
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class SemiMarkovEngine:
    """
    Semi-Markov model with regular transitions and Log-Normal state durations.

    Models discrete sequence transitions with Laplace-smoothed transition
    matrices and continuous holding times using state-conditional Log-Normal
    distributions. Computes non-conformity scores via negative log-likelihoods.

    Attributes:
        n_states (int): Total number of discrete states in the state space.
        laplace_alpha (float): Additive smoothing parameter for transition counts.
        p_hat_ (Optional[np.ndarray]): Transition probability matrix of shape
            `(n_states, n_states)`. None before `fit`.
        duration_params_ (Dict[int, Tuple[float, float]]): Mapping from state
            index to Log-Normal parameters `(mu, sigma)`.
        is_fitted_ (bool): Flag indicating whether the model has been fitted.
    """

    def __init__(self, n_states: int = 5, laplace_alpha: float = 0.01):
        """
        Initializes the SemiMarkovEngine with state space and smoothing hyperparameters.

        Args:
            n_states (int, optional): Number of discrete states. Defaults to 5.
            laplace_alpha (float, optional): Additive pseudo-count for Laplace
                smoothing. Defaults to 0.01.
        """
        self.n_states = int(n_states)
        self.laplace_alpha = float(laplace_alpha)

        self.p_hat_: Optional[np.ndarray] = None
        self.duration_params_: Dict[int, Tuple[float, float]] = {}
        self.is_fitted_: bool = False


    @property
    def P_hat_(self) -> Optional[np.ndarray]:
        """
        Backward-compatible property for transition probability matrix.
        """
        return self.p_hat_


    @P_hat_.setter
    def P_hat_(self, value: Optional[np.ndarray]) -> None:
        self.p_hat_ = value


    def fit(self, train_sequences: Sequence[Tuple[Sequence[int], Sequence[float]]]) -> "SemiMarkovEngine":
        """
        Fits transition probabilities and state duration distributions from training sequences.

        Estimates the empirical transition matrix with Laplace smoothing and fits
        state-conditional Log-Normal parameters `(mu, sigma)` using maximum
        likelihood estimation on observed holding durations.

        Args:
            train_sequences (Sequence[Tuple[Sequence[int], Sequence[float]]]):
                Collection of sequence pairs, where each item is a tuple `(events, durations)`:
                - `events`: Sequence of integer state indices of length `T + 1`.
                - `durations`: Sequence of positive transition holding times of length `T`.

        Returns:
            SemiMarkovEngine: The fitted model instance (`self`).
        """
        counts = np.full((self.n_states, self.n_states), self.laplace_alpha, dtype=float)
        durations_by_state: Dict[int, List[float]] = {i: [] for i in range(self.n_states)}

        for item in train_sequences:
            evs, durs = item[0], item[1]

            for t in range(len(durs)):
                u, v = int(evs[t]), int(evs[t + 1])
                dt = float(durs[t])
                counts[u, v] += 1.0
                durations_by_state[u].append(dt)

        self.p_hat_ = counts / counts.sum(axis=1, keepdims=True)

        self.duration_params_ = {}
        for u in range(self.n_states):
            d_list = durations_by_state[u]

            if len(d_list) > 1:
                log_d = np.log(np.maximum(1e-6, d_list))
                mu = float(np.mean(log_d))
                sigma = max(0.1, float(np.std(log_d, ddof=1)))
                self.duration_params_[u] = (mu, sigma)
            else:
                self.duration_params_[u] = (2.0, 1.0)

        self.is_fitted_ = True

        return self

    def score(
        self, sequences: Sequence[Tuple[Sequence[int], Sequence[float]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates transition and duration non-conformity scores for sequences.

        Evaluates the mean negative log-likelihood (NLL) for discrete Markovian
        state transitions and continuous Log-Normal holding times.

        Args:
            sequences (Sequence[Tuple[Sequence[int], Sequence[float]]]):
                Collection of sequence tuples `(events, durations)` to evaluate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - **trans_nlls** (`np.ndarray`): 1D array of mean negative log-likelihoods
                  for discrete state transitions across each sequence.
                - **dur_nlls** (`np.ndarray`): 1D array of mean negative log-likelihoods
                  for continuous state holding durations across each sequence.

        Raises:
            RuntimeError: If called before fitting the engine with `fit`.
        """
        if not self.is_fitted_ or self.p_hat_ is None:
            raise RuntimeError("The semi-Markovian model must be fitted using .fit() before scoring.")

        trans_nlls = []
        dur_nlls = []

        for item in sequences:
            evs, durs = item[0], item[1]
            if len(durs) == 0:
                trans_nlls.append(0.0)
                dur_nlls.append(0.0)
                continue

            log_p_trans = [
                np.log(max(1e-9, self.p_hat_[int(evs[t]), int(evs[t + 1])]))
                for t in range(len(durs))
            ]

            trans_nlls.append(-float(np.mean(log_p_trans)))

            log_p_durs = []
            for t in range(len(durs)):
                u = int(evs[t])
                dt = max(1e-6, float(durs[t]))
                mu, sigma = self.duration_params_[u]
                log_dt = np.log(dt)
                ll = (
                    -0.5 * np.log(2.0 * np.pi)
                    - np.log(sigma)
                    - log_dt
                    - ((log_dt - mu) ** 2) / (2.0 * (sigma ** 2))
                )

                log_p_durs.append(ll)

            dur_nlls.append(-float(np.mean(log_p_durs)))

        return np.asarray(trans_nlls, dtype=float), np.asarray(dur_nlls, dtype=float)

    def to_dict(self) -> dict:
        """
        Serializes the engine parameters and state into a dictionary.

        Returns:
            dict: Dictionary containing `n_states`, `laplace_alpha`, `p_hat`,
                `duration_params`, and `is_fitted`.
        """
        return {
            "n_states": self.n_states,
            "laplace_alpha": self.laplace_alpha,
            "p_hat": self.p_hat_.tolist() if self.p_hat_ is not None else None,
            "duration_params": {str(k): list(v) for k, v in self.duration_params_.items()},
            "is_fitted": self.is_fitted_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemiMarkovEngine":
        """
        Reconstructs a SemiMarkovEngine instance from a serialized dictionary.

        Args:
            data (dict): Dictionary with keys `'n_states'`, `'laplace_alpha'`,
                `'p_hat'` (or legacy `'P_hat'`), `'duration_params'`, and `'is_fitted'`.

        Returns:
            SemiMarkovEngine: A restored instance with deserialized transition
                matrix and duration parameters.
        """
        inst = cls(n_states=data["n_states"], laplace_alpha=data["laplace_alpha"])
        raw_p = data.get("p_hat", data.get("P_hat"))
        inst.p_hat_ = np.array(raw_p, dtype=float) if raw_p is not None else None
        inst.duration_params_ = {int(k): tuple(v) for k, v in data["duration_params"].items()}
        inst.is_fitted_ = bool(data["is_fitted"])

        return inst