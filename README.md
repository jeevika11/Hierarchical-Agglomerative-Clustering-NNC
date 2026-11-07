# Task 13: Hierarchical Agglomerative Clustering with Dynamic Nearest-Neighbor Chain from Scratch

## 1. Overview

This project implements Hierarchical Agglomerative Clustering (HAC) from scratch using Python and NumPy.

It calculates pairwise Euclidean distances, maintains dynamic cluster relationships, performs cluster merging using Complete Linkage and Ward's Linkage, and generates custom text-based dendrograms.

SciPy Spatial is used only for independent distance verification. Scikit-learn is not used.

## 2. Objectives

- Calculate pairwise Euclidean distance matrices.
- Implement Complete Linkage clustering.
- Implement Ward's Linkage clustering.
- Develop a dynamic nearest-neighbor chain algorithm.
- Maintain cluster relationships and merge history.
- Generate custom text-based dendrogram trees.
- Verify distance calculations using SciPy Spatial.

## 3. Technologies Used

- Python 3.10+
- NumPy
- SciPy Spatial (verification only)
- Matplotlib (dependency included for the task environment)

## 4. Project Structure

```text
Task 13/
│
├── distance_matrix.py
├── linkage.py
├── nearest_neighbor.py
├── hierarchical_clustering.py
├── dendrogram.py
├── task13.py
├── requirements.txt
└── README.md
```

## 5. Implementation Details

### Pairwise Distance Matrix

Calculates Euclidean distances between all observations using vectorized NumPy operations.

### Complete Linkage

Uses the maximum pairwise distance between observations in two clusters.

### Ward's Linkage

Calculates the increase in within-cluster sum of squares when two clusters are merged.

### Dynamic Nearest-Neighbor Chain

Maintains a chain of nearest neighboring clusters and merges mutually nearest neighbors. Distances are updated after cluster merges.

### Hierarchical Agglomerative Clustering

Starts with individual observations as clusters and repeatedly merges clusters until the requested number of clusters is reached.

### Custom Dendrogram

Generates a text-based hierarchical tree showing parent-child relationships and merge distances without scikit-learn.

## 6. Installation

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## 7. Run the Project

Run individual modules:

```bash
python distance_matrix.py
python linkage.py
python nearest_neighbor.py
python hierarchical_clustering.py
python dendrogram.py
```

Run the complete project:

```bash
python task13.py
```

## 8. Verification Results

The complete implementation was tested using a six-observation dataset.

| Verification | Result |
|---|---|
| Dataset dimensions | PASS |
| Pairwise distance calculation | PASS |
| Distance matrix symmetry | PASS |
| Complete linkage calculation | PASS |
| Ward linkage calculation | PASS |
| Complete linkage merge count | PASS |
| Ward linkage merge count | PASS |
| Complete clustering | PASS |
| Ward clustering | PASS |
| Dendrogram generation | PASS |

**Final Result: 10/10 tests passed.**

## 9. Conclusion

The project successfully implements hierarchical agglomerative clustering using Complete and Ward's Linkage with a dynamic nearest-neighbor chain. It also provides custom text-based dendrogram generation and independent verification of Euclidean distance calculations using SciPy Spatial.
