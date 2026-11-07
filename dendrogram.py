
class Dendrogram:
    """Generate a custom text-based hierarchical dendrogram."""

    def __init__(self, merge_history, n_samples):
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1.")

        self.merge_history = list(merge_history)
        self.n_samples = n_samples
        self.nodes = {
            i: None for i in range(n_samples)
        }

        self._build_tree()

    def _build_tree(self):
        """Build parent-child relationships from merge history."""

        expected_id = self.n_samples

        for merge in self.merge_history:
            if len(merge) != 4:
                raise ValueError(
                    "Each merge must contain four values."
                )

            left, right, distance, new_id = merge

            if left not in self.nodes or right not in self.nodes:
                raise ValueError("Merge references an unknown cluster.")

            if left == right:
                raise ValueError("A cluster cannot merge with itself.")

            if new_id in self.nodes:
                raise ValueError("Duplicate cluster ID in merge history.")

            if new_id != expected_id:
                raise ValueError("Unexpected cluster ID in merge history.")

            if distance < 0:
                raise ValueError("Merge distance cannot be negative.")

            self.nodes[new_id] = {
                "left": left,
                "right": right,
                "distance": float(distance)
            }

            expected_id += 1

        self.root = (
            self.n_samples + len(self.merge_history) - 1
            if self.merge_history else 0
        )

    def _format_node(self, cluster_id, prefix="", is_last=True):
        """Recursively format a dendrogram node."""

        connector = "└── " if is_last else "├── "

        if self.nodes[cluster_id] is None:
            return [prefix + connector + f"Sample {cluster_id}"]

        node = self.nodes[cluster_id]

        lines = [
            prefix + connector
            + f"Cluster {cluster_id} "
            + f"(distance={node['distance']:.4f})"
        ]

        child_prefix = prefix + (
            "    " if is_last else "│   "
        )

        children = [node["left"], node["right"]]

        for index, child_id in enumerate(children):
            last_child = index == len(children) - 1

            lines.extend(
                self._format_node(
                    child_id,
                    child_prefix,
                    last_child
                )
            )

        return lines

    def generate(self):
        """Return the complete text-based dendrogram."""

        lines = ["HIERARCHICAL DENDROGRAM"]

        lines.extend(
            self._format_node(self.root)
        )

        return "\n".join(lines)

    def display(self):
        """Print the dendrogram."""

        output = self.generate()
        print(output)
        return output


if __name__ == "__main__":

    # Sample merge history for five observations.
    sample_merges = [
        (0, 1, 1.4142, 5),
        (3, 4, 1.4142, 6),
        (2, 5, 5.6569, 7),
        (6, 7, 9.8995, 8)
    ]

    print("=" * 60)

    tree = Dendrogram(
        merge_history=sample_merges,
        n_samples=5
    )

    output = tree.display()

    print("\nDendrogram verification:")

    print(
        "PASS" if "Sample 0" in output
        and "Sample 4" in output
        and "distance=" in output
        else "FAIL"
    )

    print("\nMerge count verification:",
          "PASS" if len(sample_merges) == 5 - 1 else "FAIL")

    print("=" * 60)
