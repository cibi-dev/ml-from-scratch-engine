"""Tests for UnionFind disjoint-set data structure with path compression and union-by-rank."""

from __future__ import annotations

from minhash_dedup.clustering import UnionFind


class TestUnionFind:
    def test_initialization_empty_and_with_elements(self) -> None:
        uf_empty: UnionFind[str] = UnionFind()
        assert len(uf_empty) == 0
        assert uf_empty.component_count() == 0

        uf_init: UnionFind[int] = UnionFind([1, 2, 3, 4])
        assert len(uf_init) == 4
        assert uf_init.component_count() == 4

    def test_find_and_auto_registration(self) -> None:
        uf: UnionFind[str] = UnionFind()
        root = uf.find("a")
        assert root == "a"
        assert len(uf) == 1
        assert uf.is_connected("a", "a")

    def test_union_distinct_and_same_components(self) -> None:
        uf: UnionFind[int] = UnionFind([1, 2, 3])
        assert uf.union(1, 2) is True
        assert uf.is_connected(1, 2) is True
        assert uf.is_connected(1, 3) is False
        assert uf.component_count() == 2

        # Redundant union returns False
        assert uf.union(1, 2) is False
        assert uf.component_count() == 2

    def test_transitive_unions(self) -> None:
        uf: UnionFind[str] = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("d", "e")

        assert uf.is_connected("a", "c") is True
        assert uf.is_connected("a", "d") is False
        assert uf.component_count() == 2

    def test_get_components_mapping(self) -> None:
        uf: UnionFind[str] = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.add("isolated")

        components = uf.get_components()
        assert len(components) == 2
        root_abc = uf.find("a")
        root_iso = uf.find("isolated")

        assert set(components[root_abc]) == {"a", "b", "c"}
        assert set(components[root_iso]) == {"isolated"}

    def test_path_compression_depth_reduction(self) -> None:
        uf: UnionFind[int] = UnionFind()
        # Form a chain
        for i in range(10):
            uf.union(i, i + 1)

        # After find(0), path compression should point 0 directly to root
        root = uf.find(0)
        assert uf.parent[0] == root
