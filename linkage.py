
import numpy as np


def complete_linkage(X, cluster_a, cluster_b):
    """
    Calculate complete linkage distance.

    Complete linkage = maximum pairwise distance
    between points belonging to two clusters.
    """

    X = np.asarray(X, dtype=np.float64)

    cluster_a = np.asarray(cluster_a, dtype=int)
    cluster_b = np.asarray(cluster_b, dtype=int)

    if cluster_a.size == 0 or cluster_b.size == 0:
        raise ValueError("Clusters cannot be empty.")

    if np.any(cluster_a < 0) or np.any(cluster_a >= len(X)):
        raise IndexError("Cluster A contains invalid indices.")

    if np.any(cluster_b < 0) or np.any(cluster_b >= len(X)):
        raise IndexError("Cluster B contains invalid indices.")

    points_a = X[cluster_a]
    points_b = X[cluster_b]

    differences = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))

    return float(np.max(distances))


def ward_linkage(X, cluster_a, cluster_b):
    """
    Calculate Ward's linkage distance.

    Returns the square root of twice the increase
    in within-cluster sum of squared errors.
    """

    X = np.asarray(X, dtype=np.float64)

    cluster_a = np.asarray(cluster_a, dtype=int)
    cluster_b = np.asarray(cluster_b, dtype=int)

    if cluster_a.size == 0 or cluster_b.size == 0:
        raise ValueError("Clusters cannot be empty.")

    if np.any(cluster_a < 0) or np.any(cluster_a >= len(X)):
        raise IndexError("Cluster A contains invalid indices.")

    if np.any(cluster_b < 0) or np.any(cluster_b >= len(X)):
        raise IndexError("Cluster B contains invalid indices.")

    points_a = X[cluster_a]
    points_b = X[cluster_b]

    count_a = len(points_a)
    count_b = len(points_b)

    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)

    centroid_distance = np.linalg.norm(centroid_a - centroid_b)

    factor = (2.0 * count_a * count_b) / (count_a + count_b)

    return float(np.sqrt(factor) * centroid_distance)


def calculate_linkage(X, cluster_a, cluster_b, method="complete"):
    """Select the required linkage criterion."""

    if method == "complete":
        return complete_linkage(X, cluster_a, cluster_b)

    if method == "ward":
        return ward_linkage(X, cluster_a, cluster_b)

    raise ValueError("Method must be 'complete' or 'ward'.")


if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [2, 3],
        [5, 6],
        [8, 9],
        [9, 10]
    ], dtype=np.float64)

    cluster_a = [0, 1]
    cluster_b = [2, 3]

    complete_distance = complete_linkage(
        X, cluster_a, cluster_b
    )

    ward_distance = ward_linkage(
        X, cluster_a, cluster_b
    )

    print("=" * 55)
    print("LINKAGE CRITERIA TEST")
    print("=" * 55)

    print("Cluster A:", cluster_a)
    print("Cluster B:", cluster_b)

    print("\nComplete linkage distance:")
    print(round(complete_distance, 4))

    print("\nWard linkage distance:")
    print(round(ward_distance, 4))

    print("\nLinkage selection test:")
    print(
        "PASS" if calculate_linkage(
            X, cluster_a, cluster_b, "complete"
        ) == complete_distance else "FAIL"
    )

    print("\nPositive distance verification:",
          "PASS" if complete_distance > 0 and ward_distance > 0
          else "FAIL")

    print("\nLinkage output verification:",
          "PASS" if np.isfinite(complete_distance)
          and np.isfinite(ward_distance) else "FAIL")

    print("=" * 55)
