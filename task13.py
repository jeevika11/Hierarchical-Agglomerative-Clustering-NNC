
import numpy as np
from scipy.spatial.distance import cdist

from distance_matrix import pairwise_distance_matrix
from linkage import complete_linkage, ward_linkage
from nearest_neighbor import nearest_neighbor_chain
from hierarchical_clustering import HierarchicalClustering
from dendrogram import Dendrogram


def run_test(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{name:<38}: {status}")
    return bool(condition)


def main():
    print("=" * 65)
    print("TASK 13: HIERARCHICAL AGGLOMERATIVE CLUSTERING")
    print("DYNAMIC NEAREST-NEIGHBOR CHAIN")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Sample dataset
    # --------------------------------------------------
    X = np.array([
        [1, 2],
        [2, 3],
        [5, 6],
        [8, 9],
        [9, 10],
        [10, 11]
    ], dtype=np.float64)

    print("\nDATASET INFORMATION")
    print("-" * 65)
    print("Number of samples:", len(X))
    print("Number of features:", X.shape[1])
    print("\nInput dataset:")
    print(X)

    # --------------------------------------------------
    # 2. Pairwise distance calculation
    # --------------------------------------------------
    distances = pairwise_distance_matrix(X)
    scipy_distances = cdist(X, X, metric="euclidean")

    # --------------------------------------------------
    # 3. Linkage calculations
    # --------------------------------------------------
    cluster_a = [0, 1]
    cluster_b = [2, 3]

    complete_distance = complete_linkage(
        X, cluster_a, cluster_b
    )

    ward_distance = ward_linkage(
        X, cluster_a, cluster_b
    )

    print("\nLINKAGE DISTANCES")
    print("-" * 65)
    print("Complete linkage:", round(complete_distance, 4))
    print("Ward linkage:", round(ward_distance, 4))

    # --------------------------------------------------
    # 4. Run nearest-neighbor chain
    # --------------------------------------------------
    complete_merges = nearest_neighbor_chain(
        X, method="complete"
    )

    ward_merges = nearest_neighbor_chain(
        X, method="ward"
    )

    # --------------------------------------------------
    # 5. Hierarchical clustering
    # --------------------------------------------------
    complete_model = HierarchicalClustering(
        n_clusters=2,
        linkage="complete"
    )

    ward_model = HierarchicalClustering(
        n_clusters=2,
        linkage="ward"
    )

    complete_labels = complete_model.fit_predict(X)
    ward_labels = ward_model.fit_predict(X)

    print("\nCOMPLETE LINKAGE CLUSTERING")
    print("-" * 65)
    print("Labels:", complete_labels)
    print("Clusters:", complete_model.get_clusters())
    print("Total clusters:", len(np.unique(complete_labels)))

    print("\nWARD LINKAGE CLUSTERING")
    print("-" * 65)
    print("Labels:", ward_labels)
    print("Clusters:", ward_model.get_clusters())
    print("Total clusters:", len(np.unique(ward_labels)))

    # --------------------------------------------------
    # 6. Generate dendrograms
    # --------------------------------------------------
    print("\nCOMPLETE LINKAGE DENDROGRAM")
    print("-" * 65)

    complete_tree = Dendrogram(
        complete_merges, len(X)
    )

    complete_output = complete_tree.display()

    print("\nWARD LINKAGE DENDROGRAM")
    print("-" * 65)

    ward_tree = Dendrogram(
        ward_merges, len(X)
    )

    ward_output = ward_tree.display()

    # --------------------------------------------------
    # 7. Final verification tests
    # --------------------------------------------------
    print("\nFINAL VERIFICATION TESTS")
    print("-" * 65)

    tests = []

    tests.append(run_test(
        "Dataset dimensions",
        X.shape == (6, 2)
    ))

    tests.append(run_test(
        "Pairwise distance calculation",
        np.allclose(distances, scipy_distances)
    ))

    tests.append(run_test(
        "Distance matrix symmetry",
        np.allclose(distances, distances.T)
    ))

    tests.append(run_test(
        "Complete linkage calculation",
        np.isfinite(complete_distance)
        and complete_distance > 0
    ))

    tests.append(run_test(
        "Ward linkage calculation",
        np.isfinite(ward_distance)
        and ward_distance > 0
    ))

    tests.append(run_test(
        "Complete linkage merge count",
        len(complete_merges) == len(X) - 1
    ))

    tests.append(run_test(
        "Ward linkage merge count",
        len(ward_merges) == len(X) - 1
    ))

    tests.append(run_test(
        "Complete clustering",
        len(np.unique(complete_labels)) == 2
        and len(complete_labels) == len(X)
    ))

    tests.append(run_test(
        "Ward clustering",
        len(np.unique(ward_labels)) == 2
        and len(ward_labels) == len(X)
    ))

    tests.append(run_test(
        "Dendrogram generation",
        "Sample 0" in complete_output
        and "Sample 5" in complete_output
        and "Sample 0" in ward_output
        and "Sample 5" in ward_output
    ))

    # --------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------
    passed = sum(tests)
    total = len(tests)

    print("\n" + "=" * 65)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 65)

    print(f"Tests passed: {passed}/{total}")
    print(f"Tests failed: {total - passed}/{total}")

    if passed == total:
        print("\nTASK 13 STATUS: ALL TESTS PASSED")
    else:
        print("\nTASK 13 STATUS: CHECK FAILED TESTS")

    print("=" * 65)


if __name__ == "__main__":
    main()
