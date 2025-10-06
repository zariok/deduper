from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar


ValueT = TypeVar("ValueT")
PayloadT = TypeVar("PayloadT")


class _BKNode(Generic[ValueT, PayloadT]):
    """A single node within a BK-tree."""

    __slots__ = ("value", "payloads", "children")

    def __init__(self, value: ValueT, payload: PayloadT):
        self.value: ValueT = value
        self.payloads: List[PayloadT] = [payload]
        self.children: Dict[int, "_BKNode[ValueT, PayloadT]"] = {}

    def add(self, value: ValueT, payload: PayloadT, distance_func: Callable[[ValueT, ValueT], int]) -> None:
        distance = distance_func(value, self.value)

        if distance == 0:
            if payload not in self.payloads:
                self.payloads.append(payload)
            return

        child = self.children.get(distance)
        if child is None:
            self.children[distance] = _BKNode(value, payload)
            return

        child.add(value, payload, distance_func)

    def search(self, value: ValueT, tolerance: int, distance_func: Callable[[ValueT, ValueT], int], results: List[Tuple[PayloadT, int]]) -> None:
        distance = distance_func(value, self.value)

        if distance <= tolerance:
            for payload in self.payloads:
                results.append((payload, distance))

        lower = max(distance - tolerance, 1)
        upper = distance + tolerance

        for child_distance, child in self.children.items():
            if lower <= child_distance <= upper:
                child.search(value, tolerance, distance_func, results)


class BKTree(Generic[ValueT, PayloadT]):
    """A metric tree that supports efficient nearest-neighbour queries within a bounded distance."""

    def __init__(self, distance_func: Callable[[ValueT, ValueT], int]):
        self._distance = distance_func
        self._root: Optional[_BKNode[ValueT, PayloadT]] = None

    def add(self, value: ValueT, payload: PayloadT) -> None:
        if self._root is None:
            self._root = _BKNode(value, payload)
            return

        self._root.add(value, payload, self._distance)

    def bulk_add(self, items: Iterable[Tuple[ValueT, PayloadT]]) -> None:
        for value, payload in items:
            self.add(value, payload)

    def search(self, value: ValueT, tolerance: int) -> List[Tuple[PayloadT, int]]:
        if self._root is None:
            return []

        results: List[Tuple[PayloadT, int]] = []
        self._root.search(value, tolerance, self._distance, results)
        return results