"""Ground-truth protocols for frame-aligned and metric-distance benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def _normalise(path: str) -> str:
    return Path(path).as_posix().removeprefix("./")


@dataclass(frozen=True)
class PairManifestGroundTruth:
    """Valid query/database pairs loaded from a metric-distance CSV manifest.

    The CSV must contain ``query_path``, ``database_path`` and ``distance_m``.
    Paths use the same relative strings stored in the feature artifacts. Rows
    beyond ``distance_threshold_m`` are ignored.
    """

    matches: dict[str, frozenset[str]]
    distance_threshold_m: float

    @classmethod
    def from_csv(
        cls, manifest_path: str | Path, distance_threshold_m: float
    ) -> "PairManifestGroundTruth":
        matches: dict[str, set[str]] = {}
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"query_path", "database_path", "distance_m"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"ground-truth CSV needs columns {sorted(required)}, "
                    f"got {reader.fieldnames}"
                )
            for row in reader:
                if float(row["distance_m"]) <= distance_threshold_m:
                    query = _normalise(row["query_path"])
                    database = _normalise(row["database_path"])
                    matches.setdefault(query, set()).add(database)
        return cls(
            matches={query: frozenset(paths) for query, paths in matches.items()},
            distance_threshold_m=distance_threshold_m,
        )

    def is_correct(self, query_path: str, database_path: str) -> bool:
        return _normalise(database_path) in self.matches.get(
            _normalise(query_path), frozenset()
        )

    def has_match(self, query_path: str, database_paths: list[str]) -> bool:
        valid = self.matches.get(_normalise(query_path), frozenset())
        return any(_normalise(path) in valid for path in database_paths)

