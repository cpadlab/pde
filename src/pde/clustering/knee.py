
from typing import List, Sequence, Tuple
import numpy as np


class KneeLocator:
    """
    Geometric knee point locator compatible with NumPy 2.0+.

    Finds the optimal elbow/knee point on an inertia curve by computing
    the perpendicular distance from each evaluated point to the secant line
    connecting the first and last points.
    """

    @staticmethod
    def detect_knee_point(k_values: Sequence[int], inertias: Sequence[float]) -> Tuple[int, List[float]]:
        """
        Calculates the optimal knee point on the inertia curve.

        Computes the perpendicular distance from each `(k, inertia)` point to
        the secant line drawn between the first and last evaluated points,
        selecting the point with the maximum orthogonal distance.

        Args:
            k_values (Sequence[int]): Evaluated values of k (cluster counts)
                in strictly monotonic order.
            inertias (Sequence[float]): Corresponding inertia values (sum of
                squared or absolute distances) for each k.

        Returns:
            Tuple[int, List[float]]: A tuple containing:
                - **optimal_k** (`int`): The selected optimal value of k at
                  the knee point.
                - **orthogonal_distances** (`List[float]`): List of computed
                  orthogonal distances to the secant line for each evaluated point.

        Raises:
            ValueError: If fewer than 3 values of k are provided.
        """
        ks = np.asarray(k_values, dtype=float)
        ins = np.asarray(inertias, dtype=float)

        if len(ks) < 3:
            raise ValueError("At least 3 values of k are required to determine a knee.")

        x1, y1 = ks[0], ins[0]
        x2, y2 = ks[-1], ins[-1]

        dx = x2 - x1
        dy = y2 - y1
        denom = np.sqrt(dx**2 + dy**2)

        distances = []
        for x0, y0 in zip(ks, ins):
            dist = np.abs(dx * (y1 - y0) - dy * (x1 - x0)) / denom
            distances.append(float(dist))

        best_idx = int(np.argmax(distances))
        return int(ks[best_idx]), distances