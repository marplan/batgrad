from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pyarrow.parquet as pq

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId, MappingSpec
from batgrad.data.processing.raw import IngestBatch, IngestStageSpec, IngestTask, plan_file_tasks

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.storage.store import DataProcessingStore


@dataclass(frozen=True)
class SyntheticPozzato2022RawAdapter:
    spec: DatasetSpec

    FOOTER_KEY: ClassVar[bytes] = b"hub_data.file"

    def plan_raw_tasks(
        self,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> tuple[IngestTask, ...]:
        raw_root = self.spec.source_root(DatasetStageId.raw)
        return plan_file_tasks(input_store, raw_root, raw_spec)

    def load_raw_task(
        self,
        task: IngestTask,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> Iterator[IngestBatch]:
        source_path = task.source_paths[0]
        metadata = self._extract_metadata(source_path, input_store)
        protocol_id = metadata[BaseColumns.proto]
        if not isinstance(protocol_id, DatasetProtocolId):
            protocol_id = DatasetProtocolId(str(protocol_id))
        raw_spec.protocol_spec(protocol_id)

        yield IngestBatch(
            data=input_store.scan_table(source_path),
            protocol_id=protocol_id,
            source_paths=(source_path,),
            metadata=metadata,
        )

    def _extract_metadata(
        self,
        source_path: str,
        input_store: DataProcessingStore,
    ) -> dict[MappingSpec, object]:
        footer = self._read_hub_footer(source_path, input_store)
        index_raw = footer.get("index", {})
        index: dict[str, object] = (
            {str(key): value for key, value in index_raw.items()}
            if isinstance(index_raw, dict)
            else {}
        )
        protocol_id = self._infer_protocol(source_path, index.get("domain"))
        metadata: dict[MappingSpec, object] = {
            BaseColumns.proto: protocol_id,
            BaseColumns.cell_id: self._required_str(index, "cell_id", source_path),
            BaseColumns.cidx: self._required_int(index, "cycle_index", source_path),
        }
        if "nominal_capacity_ah" in index:
            metadata[BaseColumns.cap_nom] = self._numeric_value(
                index["nominal_capacity_ah"],
                "nominal_capacity_ah",
                source_path,
            )
        if protocol_id is DatasetProtocolId.eis:
            metadata[BaseColumns.soc_v] = self._extract_eis_voltage(source_path)
        return metadata

    def _read_hub_footer(
        self,
        source_path: str,
        input_store: DataProcessingStore,
    ) -> dict[str, object]:
        with input_store.local_file(source_path) as path:
            raw_metadata = pq.ParquetFile(path).metadata.metadata or {}
        raw_payload = raw_metadata.get(self.FOOTER_KEY)
        if raw_payload is None:
            raise ValueError(f"Synthetic source parquet is missing hub metadata: {source_path}")
        payload = json.loads(raw_payload.decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Synthetic source parquet has invalid hub metadata: {source_path}")
        return payload

    @staticmethod
    def _infer_protocol(source_path: str, domain: object) -> DatasetProtocolId:
        stem = Path(source_path).stem
        domain_value = str(domain) if domain is not None else ""
        if stem.startswith("Cycling_") or domain_value == "Cycling":
            return DatasetProtocolId.cycling
        if stem.startswith("Diag_") or domain_value == "Diag":
            return DatasetProtocolId.rpt
        if stem.startswith("EIS") or domain_value == "EIS":
            return DatasetProtocolId.eis
        raise ValueError(f"Could not infer synthetic protocol from source path: {source_path}")

    @staticmethod
    def _extract_eis_voltage(source_path: str) -> float:
        match = re.search(r"EIS(\d+(?:p\d+|\.\d+)?)V", Path(source_path).stem)
        if match is None:
            raise ValueError(f"Could not infer EIS voltage from source path: {source_path}")
        return float(match.group(1).replace("p", "."))

    @staticmethod
    def _required_str(index: dict[str, object], key: str, source_path: str) -> str:
        value = index.get(key)
        if value is None:
            raise ValueError(f"Synthetic source {source_path} is missing footer index {key!r}")
        return str(value)

    @staticmethod
    def _required_int(index: dict[str, object], key: str, source_path: str) -> int:
        value = index.get(key)
        if value is None:
            raise ValueError(f"Synthetic source {source_path} is missing footer index {key!r}")
        return int(SyntheticPozzato2022RawAdapter._numeric_value(value, key, source_path))

    @staticmethod
    def _numeric_value(value: object, key: str, source_path: str) -> float:
        if isinstance(value, int | float | str):
            return float(value)
        raise TypeError(
            f"Synthetic source {source_path} has non-numeric footer index {key!r}: {value!r}"
        )
