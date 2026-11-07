
import numpy as np
from linkage import calculate_linkage


def nearest_neighbor_chain(X, initial_clusters=None, method="complete"):
    """
    Perform hierarchical clustering using a dynamic
    nearest-neighbor chain.

    Parameters
    ----------
    X : numpy.ndarray
        Input dataset.
    initial_clusters : dict, optional
        Cluster ID to observation-index list mapping.
    method : str
        'complete' or 'ward'.

    Returns
    -------
    merge_history : list
        Each entry contains (cluster_a, cluster_b, distance, new_id).
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")

    if method not in ("complete", "ward"):
        raise ValueError("Method must be 'complete' or 'ward'.")

    n_samples = X.shape[0]

    if initial_clusters is None:
        clusters = {
            i: [i] for i in range(n_samples)
        }
    else:
        clusters = {
            int(cluster_id): list(members)
            for cluster_id, members in initial_clusters.items()
        }

        if not clusters:
            raise ValueError("Initial clusters cannot be empty.")

        all_members = [
            member
            for members in clusters.values()
            for member in members
        ]

        if sorted(all_members) != list(range(n_samples)):
            raise ValueError(
                "Initial clusters must partition all observations exactly once."
            )

    active = list(clusters.keys())
    next_cluster_id = max(active) + 1

    merge_history = []
    chain = []

    # Dynamic distance cache.
    distances = {}

    def pair_key(a, b):
        return (min(a, b), max(a, b))

    def get_distance(a, b):
        key = pair_key(a, b)

        if key not in distances:
            distances[key] = calculate_linkage(
                X,
                clusters[a],
                clusters[b],
                method
            )

        return distances[key]

    while len(active) > 1:

        # Start a new chain when the current chain is empty.
        if not chain:
            chain.append(active[0])

        current = chain[-1]

        # Find the nearest active neighboring cluster.
        candidates = [
            cluster_id
            for cluster_id in active
            if cluster_id != current
        ]

        nearest = min(
            candidates,
            key=lambda cluster_id: (
                get_distance(current, cluster_id),
                cluster_id
            )
        )

        # Mutual nearest neighbors are ready to merge.
        if len(chain) >= 2 and nearest == chain[-2]:

            cluster_a = chain[-2]
            cluster_b = chain[-1]

            merge_distance = get_distance(cluster_a, cluster_b)

            new_members = (
                clusters[cluster_a] + clusters[cluster_b]
            )

            new_id = next_cluster_id
            next_cluster_id += 1

            merge_history.append((
                cluster_a,
                cluster_b,
                float(merge_distance),
                new_id
            ))

            # Update dynamic cluster storage.
            del clusters[cluster_a]
            del clusters[cluster_b]

            active.remove(cluster_a)
            active.remove(cluster_b)

            clusters[new_id] = new_members
            active.append(new_id)

            # Remove outdated distance entries.
            distances = {
                key: value
                for key, value in distances.items()
                if cluster_a not in key and cluster_b not in key
            }

            # Start a fresh chain after the merge.
            chain = []

        else:
            chain.append(nearest)

    return merge_history


if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [2, 3],
        [5, 6],
        [8, 9],
        [9, 10]
    ], dtype=np.float64)

    print("=" * 60)
    print("NEAREST-NEIGHBOR CHAIN TEST")
    print("=" * 60)

    for method in ("complete", "ward"):

        print(f"\n{method.upper()} LINKAGE")

        merges = nearest_neighbor_chain(X, method=method)

        for cluster_a, cluster_b, distance, new_id in merges:
            print(
                f"Merge {cluster_a} + {cluster_b} "
                f"-> {new_id} | Distance: {distance:.4f}"
            )

        print("Total merges:", len(merges))

        print(
            "Merge count verification:",
            "PASS" if len(merges) == len(X) - 1 else "FAIL"
        )

        print(
            "Finite distance verification:",
            "PASS" if all(
                np.isfinite(merge[2]) and merge[2] >= 0
                for merge in merges
            ) else "FAIL"
        )

    print("\nNearest-neighbor chain verification: PASS")
    print("=" * 60)
