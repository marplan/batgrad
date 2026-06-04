from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DatasetType = Literal["published", "synthetic"]
DatasetSource = Literal["raw", "parquet", "normalized"]


@dataclass(frozen=True, slots=True)
class DatasetLocation:
    dataset_type: DatasetType
    dataset_id: str
    root_overwrite: str | None = None

    def root(self) -> str:
        """Relative root location of the dataset.

        Absolute root location will be resolved by `DataStore`.
        """
        if self.root_overwrite is not None:
            return self.root_overwrite
        return f"type={self.dataset_type}/dataset={self.dataset_id}"

    def source_root(self, source: DatasetSource) -> str:
        return f"{self.root()}/source={source}"

    def source_file(self, source: DatasetSource, file_name: str) -> str:
        return f"{self.source_root(source)}/{file_name}"

    def manifest(self, source: Literal["parquet", "normalized"]) -> str:
        return self.source_file(source, "manifest.parquet")
