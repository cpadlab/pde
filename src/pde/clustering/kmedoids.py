
from typing import Optional, Tuple
import numpy as np


class RobustKMedoids:
    """
    k-Medoids clustering on precomputed distance matrices with greedy++ initialization.

    Optimizes cluster medoids using an alternating Voronoi iteration algorithm
    initialized via a k-medoids++ probabilistic scheme. Operates directly on
    precomputed pairwise distance matrices.

    Attributes:
        n_clusters (int): Number of clusters to form and medoids to generate.
        n_init (int): Number of times the algorithm will be run with different
            medoid seeds.
        max_iter (int): Maximum number of iterations for a single run.
        random_state (Optional[int]): Seed used by the random number generator.
        labels_ (Optional[np.ndarray]): Labels of each point from the best run.
            Shape `(n_samples,)`. None before `fit`.
        medoid_indices_ (Optional[np.ndarray]): Indices of the medoids in the
            training distance matrix from the best run. Shape `(n_clusters,)`.
            None before `fit`.
        inertia_ (float): Sum of distances of samples to their closest medoid
            from the best run. Set to infinity before `fit`.
    """


    def __init__(
        self, n_clusters: int = 3, n_init: int = 10, 
        max_iter: int = 100, random_state: Optional[int] = None
    ):
        """
        Initializes the RobustKMedoids model with hyperparameters.

        Args:
            n_clusters (int, optional): The number of clusters to form and
                medoids to generate. Defaults to 3.
            n_init (int, optional): Number of consecutive initializations to
                perform. The best output in terms of inertia is kept. Defaults to 10.
            max_iter (int, optional): Maximum number of iterations for a single
                optimization run. Defaults to 100.
            random_state (Optional[int], optional): Random seed to control medoid
                selection randomness during initialization. Defaults to None.
        """
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = random_state

        self.labels_: Optional[np.ndarray] = None
        self.medoid_indices_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")


    @staticmethod
    def _k_medoids_plus_plus_init(
        dist_matrix: np.ndarray, n_clusters: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Selects initial cluster medoids using the k-medoids++ sampling strategy.

        Args:
            dist_matrix (np.ndarray): Square pairwise distance matrix of shape
                `(n_samples, n_samples)`.
            n_clusters (int): Number of medoid indices to select.
            rng (np.random.Generator): NumPy random number generator instance.

        Returns:
            np.ndarray: 1D array of shape `(n_clusters,)` containing the chosen
                medoid indices.
        """
        n = dist_matrix.shape[0]
        medoids = [int(rng.integers(0, n))]

        for _ in range(1, n_clusters):
            min_dists = np.min(dist_matrix[:, medoids], axis=1)
            probs = min_dists ** 2
            p_sum = probs.sum()
            probs = probs / p_sum if p_sum > 0 else np.full(n, 1.0 / n)
            next_medoid = int(rng.choice(n, p=probs))
            medoids.append(next_medoid)

        return np.array(medoids, dtype=int)


    def fit(self, dist_matrix: np.ndarray) -> "RobustKMedoids":
        """
        Fits k-Medoids clustering from a precomputed distance matrix.

        Performs `n_init` runs of k-medoids alternating minimization on
        the precomputed distance matrix and retains the solution minimizing
        overall cluster inertia.

        Args:
            dist_matrix (np.ndarray): Square pairwise distance matrix of shape
                `(n_samples, n_samples)`.

        Returns:
            RobustKMedoids: The fitted estimator instance (`self`).
        """
        rng = np.random.default_rng(self.random_state)
        best_inertia = float("inf")
        best_labels = None
        best_medoids = None

        for _ in range(self.n_init):
            medoid_indices = self._k_medoids_plus_plus_init(dist_matrix, self.n_clusters, rng)

            for _it in range(self.max_iter):
                dists_to_medoids = dist_matrix[:, medoid_indices]
                labels = np.argmin(dists_to_medoids, axis=1)

                new_medoids = medoid_indices.copy()
                for c in range(self.n_clusters):
                    
                    cluster_pts = np.nonzero(labels == c)[0]
                    if len(cluster_pts) == 0:
                        continue
                    
                    sub_dists = dist_matrix[np.ix_(cluster_pts, cluster_pts)]
                    best_point = cluster_pts[np.argmin(sub_dists.sum(axis=1))]
                    new_medoids[c] = best_point

                if np.array_equal(new_medoids, medoid_indices):
                    break

                medoid_indices = new_medoids

            dists_to_medoids = dist_matrix[:, medoid_indices]
            labels = np.argmin(dists_to_medoids, axis=1)
            inertia = float(np.sum(np.min(dists_to_medoids, axis=1)))

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_medoids = medoid_indices

        self.labels_ = best_labels
        self.medoid_indices_ = best_medoids
        self.inertia_ = best_inertia

        return self


    def predict(self, dist_matrix_to_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inductively assigns new observations to the nearest training medoid.

        Args:
            dist_matrix_to_train (np.ndarray): Pairwise distance matrix of shape
                `(n_query_samples, n_train_samples)` between query samples and
                training observations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - **assigned_clusters** (`np.ndarray`): 1D array of shape
                  `(n_query_samples,)` with assigned cluster indices.
                - **min_distances** (`np.ndarray`): 1D array of shape
                  `(n_query_samples,)` containing the distance to the closest
                  assigned medoid for each query observation.

        Raises:
            RuntimeError: If called before fitting the model with `fit`.
        """
        if self.medoid_indices_ is None:
            raise RuntimeError("The model must be fitted with .fit() before predicting.")

        dists_to_medoids = dist_matrix_to_train[:, self.medoid_indices_]
        assigned_clusters = np.argmin(dists_to_medoids, axis=1)
        min_distances = np.min(dists_to_medoids, axis=1)
        
        return assigned_clusters, min_distances