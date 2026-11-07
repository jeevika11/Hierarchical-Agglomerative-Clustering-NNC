
import numpy as np


def euclidean_distance(point_a, point_b):
    """Calculate Euclidean distance between two points."""

    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)

    if point_a.shape != point_b.shape:
        raise ValueError("Both points must have the same dimensions.")

    if not np.all(np.isfinite(point_a)) or not np.all(np.isfinite(point_b)):
        raise ValueError("Points must contain finite values.")

    return np.sqrt(np.sum((point_a - point_b) ** 2))


def pairwise_distance_matrix(X):
    """Calculate pairwise Euclidean distances using NumPy."""

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("Input must be a two-dimensional array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Input dataset cannot be empty.")

    if not np.all(np.isfinite(X)):
        raise ValueError("Dataset must contain finite values.")

    # Calculate squared distances using vectorized operations.
    squared_norms = np.sum(X ** 2, axis=1)

    squared_distances = (
        squared_norms[:, np.newaxis]
        + squared_norms[np.newaxis, :]
        - 2 * (X @ X.T)
    )

    # Avoid small negative values caused by floating-point precision.
    squared_distances = np.maximum(squared_distances, 0.0)

    distances = np.sqrt(squared_distances)

    # Ensure exact symmetry and zero diagonal.
    distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)

    return distances


if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [2, 3],
        [5, 6],
        [8, 9],
        [9, 10]
    ], dtype=np.float64)

    distances = pairwise_distance_matrix(X)

    print("=" * 55)
    print("PAIRWISE EUCLIDEAN DISTANCE MATRIX TEST")
    print("=" * 55)

    print("\nInput dataset:")
    print(X)

    print("\nPairwise distance matrix:")
    print(np.round(distances, 3))

    # Independent verification using SciPy.
    from scipy.spatial.distance import cdist

    scipy_distances = cdist(X, X, metric="euclidean")

    print("\nSciPy verification:")
    print(
        "PASS" if np.allclose(distances, scipy_distances)
        else "FAIL"
    )

    symmetry_check = np.allclose(distances, distances.T)
    diagonal_check = np.allclose(np.diag(distances), 0.0)
    nonnegative_check = np.all(distances >= 0)

    print("\nSymmetry verification:",
          "PASS" if symmetry_check else "FAIL")

    print("Zero diagonal verification:",
          "PASS" if diagonal_check else "FAIL")

    print("Non-negative distance verification:",
          "PASS" if nonnegative_check else "FAIL")

    print("=" * 55)
