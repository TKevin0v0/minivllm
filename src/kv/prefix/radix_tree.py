from __future__ import annotations

import time
from collections.abc import Sequence as AbcSequence


def _prefix_len(a: AbcSequence[int], b: AbcSequence[int]) -> int:
    n = 0
    m = min(len(a), len(b))
    while n < m and a[n] == b[n]:
        n += 1
    return n


class _RadixNode:
    __slots__ = ("children", "parent", "key", "value", "ref_count", "access_time")

    def __init__(
        self,
        *,
        parent: _RadixNode | None = None,
        key: tuple[int, ...] = (),
        value: tuple[int, ...] = (),
        ref_count: int = 0,
    ) -> None:
        self.children: dict[int, _RadixNode] = {}
        self.parent = parent
        self.key = key
        self.value = value
        self.ref_count = ref_count
        self.access_time = time.monotonic()


class RadixTree:
    def __init__(self) -> None:
        self.root = _RadixNode(ref_count=1)
        self._block_to_node: dict[int, _RadixNode] = {}

    def match_prefix(self, key: list[int]) -> tuple[list[int], _RadixNode]:
        if not key:
            return [], self.root
        node = self.root
        values: list[int] = []
        child_key = key[0]
        while key and child_key in node.children:
            child = node.children[child_key]
            child.access_time = time.monotonic()
            pl = _prefix_len(child.key, key)
            if pl < len(child.key):
                new_node = self._split(child, pl)
                values.extend(new_node.value)
                return values, new_node
            values.extend(child.value)
            node = child
            key = key[pl:]
            child_key = key[0] if key else -1
        return values, node

    def insert(self, key: list[int], value: list[int]) -> _RadixNode:
        if not key:
            return self.root
        node = self.root
        child_key = key[0]
        while key and child_key in node.children:
            child = node.children[child_key]
            child.access_time = time.monotonic()
            pl = _prefix_len(child.key, key)
            key = key[pl:]
            value = value[pl:]
            child_key = key[0] if key else -1
            if pl < len(child.key):
                node = self._split(child, pl)
                break
            node = child
        if key:
            return self._add(node, tuple(key), tuple(value))
        return node

    def inc_ref(self, node: _RadixNode) -> None:
        while node is not self.root:
            node.ref_count += 1
            node = node.parent or self.root

    def dec_ref(self, node: _RadixNode) -> None:
        while node is not self.root:
            node.ref_count -= 1
            node = node.parent or self.root

    def _index(self, node: _RadixNode) -> None:
        for bid in node.value:
            self._block_to_node[bid] = node

    def _unindex(self, node: _RadixNode) -> None:
        for bid in node.value:
            if self._block_to_node.get(bid) is node:
                del self._block_to_node[bid]

    def _add(
        self, parent: _RadixNode, key: tuple[int, ...], value: tuple[int, ...]
    ) -> _RadixNode:
        node = _RadixNode(parent=parent, key=key, value=value)
        parent.children[key[0]] = node
        self._index(node)
        return node

    def _split(self, node: _RadixNode, split_at: int) -> _RadixNode:
        self._unindex(node)
        new_node = _RadixNode(
            parent=node.parent,
            key=node.key[:split_at],
            value=node.value[:split_at],
            ref_count=node.ref_count,
        )
        new_node.children = {node.key[split_at]: node}
        node.parent = new_node
        node.key = node.key[split_at:]
        node.value = node.value[split_at:]
        if new_node.parent is not None and new_node.key:
            new_node.parent.children[new_node.key[0]] = new_node
        self._index(new_node)
        self._index(node)
        return new_node
