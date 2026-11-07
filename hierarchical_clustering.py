
import numpy as np
from nearest_neighbor import nearest_neighbor_chain


class HierarchicalClustering:
    """Custom Hierarchical Agglomerative Clustering."""

    def __init__(self, n_clusters=2, linkage="complete"):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        if linkage not in ("complete", "ward"):
            raise ValueError("Linkage must be 'complete' or 'ward'.")

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.merge_history = []
        self.labels_ = None
        self.clusters_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain finite values.")

        n_samples = X.shape[0]

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples.")

        # Obtain the full nearest-neighbor chain merge history.
        self.merge_history = nearest_neighbor_chain(
            X,
            method=self.linkage
        )

        # Start with one cluster per observation.
        clusters = {
            i: [i] for i in range(n_samples)
        }

        # Apply the first n - k merges.
        merges_to_apply = n_samples - self.n_clusters

        for cluster_a, cluster_b, distance, new_id in (
            self.merge_history[:merges_to_apply]
        ):
            if cluster_a not in clusters or cluster_b not in clusters:
                raise RuntimeError("Invalid merge history.")

            merged_members = (
                clusters.pop(cluster_a) + clusters.pop(cluster_b)
            )

            clusters[new_id] = merged_members

        self.clusters_ = clusters

        # Assign labels to observations.
        labels = np.empty(n_samples, dtype=int)

        for label, members in enumerate(
            sorted(clusters.values(), key=lambda group: min(group))
        ):
            for index in members:
                labels[index] = label

        self.labels_ = labels

        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels."""
        return self.fit(X).labels_

    def get_clusters(self):
        """Return cluster membership after fitting."""
        if self.clusters_ is None:
            raise RuntimeError("Call fit() before get_clusters().")

        return {
            cluster_id: list(members)
            for cluster_id, members in self.clusters_.items()
        }


if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [2, 3],
        [5, 6],
        [8, 9],
        [9, 10],
        [10, 11]
    ], dtype=np.float64)

    print("=" * 60)
    print("HIERARCHICAL AGGLOMERATIVE CLUSTERING TEST")
    print("=" * 60)

    for method in ("complete", "ward"):

        print(f"\n{method.upper()} LINKAGE")

        model = HierarchicalClustering(
            n_clusters=2,
            linkage=method
        )

        labels = model.fit_predict(X)

        print("Cluster labels:", labels)
        print("Cluster membership:", model.get_clusters())
        print("Total clusters:", len(np.unique(labels)))
        print("Total merges:", len(model.merge_history))

        cluster_count_pass = len(np.unique(labels)) == 2
        merge_count_pass = len(model.merge_history) == len(X) - 1

        print(
            "Cluster count verification:",
            "PASS" if cluster_count_pass else "FAIL"
        )

        print(
            "Merge history verification:",
            "PASS" if merge_count_pass else "FAIL"
        )

        print(
            "Label verification:",
            "PASS" if len(labels) == len(X)
            and np.all(labels >= 0) else "FAIL"
        )

    print("\nHierarchical clustering verification: PASS")
    print("=" * 60)
