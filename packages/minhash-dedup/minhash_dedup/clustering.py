"""Disjoint-Set Union-Find data structure with path compression and union-by-rank."""

from __future__ import annotations

from typing import Generic, Hashable, Iterable, TypeVar

T = TypeVar("T", bound=Hashable)


class UnionFind(Generic[T]):
    """Disjoint-set data structure with path compression and union-by-rank.

    Provides near-linear O(alpha(N)) time complexity for disjoint set operations,
    where alpha is the inverse Ackermann function.
    """

    def __init__(self, elements: Iterable[T] | None = None) -> None:
        """Initialize UnionFind structure with optional initial elements."""
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}

        if elements is not None:
            for elem in elements:
                self.add(elem)

    def add(self, x: T) -> None:
        """Add an element as a new singleton set if not already present."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: T) -> T:
        """Find the representative root of element x with full path compression.

        If x is not yet registered, it is automatically added.
        """
        if x not in self.parent:
            self.add(x)
            return x

        # Two-pass path compression (iterative or recursion)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression step
        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt

        return root

    def union(self, x: T, y: T) -> bool:
        """Merge the sets containing x and y using union-by-rank.

        Args:
            x: First element.
            y: Second element.

        Returns:
            True if elements were in distinct sets and merged, False if already in same set.
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank: attach tree with smaller rank under tree with higher rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def is_connected(self, x: T, y: T) -> bool:
        """Check whether x and y belong to the same connected component."""
        return self.find(x) == self.find(y)

    def get_components(self) -> dict[T, list[T]]:
        """Return a mapping from component root representative to list of member elements."""
        components: dict[T, list[T]] = {}
        for elem in self.parent:
            root = self.find(elem)
            components.setdefault(root, []).append(elem)
        return components

    def component_count(self) -> int:
        """Return the number of disjoint connected components."""
        return len(self.get_components())

    def __len__(self) -> int:
        """Return the total number of tracked elements."""
        return len(self.parent)
