from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import fastexcel
import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId, DatasetStageId, MappingSpec
from batgrad.data.processing.raw import (
    IngestBatch,
    IngestProtocolSpec,
    IngestStageSpec,
    IngestTask,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.data.datasets.config import DatasetSpec
    from batgrad.storage.store import DataProcessingStore


@dataclass(frozen=True)
class Pozzato2022RawAdapter:
    """Raw Excel adapter for the Pozzato 2022 dataset conventions.

    The adapter plans one task per included source file, infers protocol and task
    metadata from the source path, selects exactly one data sheet, and injects
    null columns for missing declared raw columns before generic ingest alignment.
    Path conventions are part of the adapter contract: EIS SOC is parsed from
    `SOC...` or `EIS...` tokens, cycling paths carry the cell id after
    `Cycling_N`, and `Diag_N` directories map to cycle index `N - 1`.
    """

    spec: DatasetSpec

    EIS_SHEET_PATTERN: ClassVar[str] = "ACIM"
    TIMESERIES_SHEET_PATTERN: ClassVar[str] = "channel"
    EXCLUDED_SHEET_PATTERN: ClassVar[str] = "chart"

    EIS_FILE_INDICATOR: ClassVar[str] = "EIS"
    HPPC_FILE_INDICATOR: ClassVar[str] = "HPPC"
    DIAG_FILE_INDICATOR: ClassVar[str] = "Diag"
    CYCLING_FILE_INDICATOR: ClassVar[str] = "Cycling"
    PART_INDICATOR: ClassVar[str] = "Part"

    PROTOCOL_SORT_ORDER: ClassVar[dict[DatasetProtocolId, int]] = {
        DatasetProtocolId.cycling: 0,
        DatasetProtocolId.hppc: 1,
        DatasetProtocolId.rpt: 2,
        DatasetProtocolId.eis: 3,
    }

    def plan_raw_tasks(
        self,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> tuple[IngestTask, ...]:
        raw_root = self.spec.source_root(DatasetStageId.raw)
        paths: list[str] = []
        for pattern in raw_spec.included_file_patterns:
            paths.extend(
                path
                for path in input_store.list_files(raw_root, pattern=pattern)
                if raw_spec.is_included_file(path)
            )
        return tuple(
            IngestTask(task_id=path, source_paths=(path,))
            for path in sorted(set(paths), key=self._source_sort_key)
        )

    def load_raw_task(
        self,
        task: IngestTask,
        input_store: DataProcessingStore,
        raw_spec: IngestStageSpec,
    ) -> Iterator[IngestBatch]:
        source_path = task.source_paths[0]
        metadata = self._extract_metadata(source_path)
        protocol_id = metadata[BaseColumns.proto]
        if not isinstance(protocol_id, DatasetProtocolId):
            protocol_id = DatasetProtocolId(str(protocol_id))
        raw_spec.protocol_spec(protocol_id)

        with input_store.local_file(source_path) as path:
            excel = fastexcel.read_excel(path)
            sheet_name = self._select_sheet(source_path, tuple(excel.sheet_names))
            sheet = excel.load_sheet(idx_or_name=sheet_name, dtypes="string")

        data = pl.LazyFrame(sheet)
        data = self._with_missing_declared_columns(data, raw_spec.protocol_spec(protocol_id))

        yield IngestBatch(
            data=data,
            protocol_id=protocol_id,
            source_paths=(source_path,),
            metadata=metadata,
        )

    def _with_missing_declared_columns(
        self,
        data: pl.LazyFrame,
        protocol_spec: IngestProtocolSpec,
    ) -> pl.LazyFrame:
        columns = set(data.collect_schema())
        exprs = [
            pl.lit(None, dtype=spec.dtype).alias(spec)
            for spec in protocol_spec.columns
            if spec.matching_name(columns) is None
        ]
        return data.with_columns(exprs) if exprs else data

    def _select_sheet(self, source_path: str, sheet_names: tuple[str, ...]) -> str:
        if self.EIS_FILE_INDICATOR in source_path:
            matches = [name for name in sheet_names if self.EIS_SHEET_PATTERN in name]
        else:
            matches = [
                name
                for name in sheet_names
                if self.TIMESERIES_SHEET_PATTERN in name.casefold()
                and self.EXCLUDED_SHEET_PATTERN not in name.casefold()
            ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one data sheet in {source_path}, got {matches}")
        return matches[0]

    def _extract_metadata(self, source_path: str) -> dict[MappingSpec, object]:
        protocol_id, soc_pct = self._infer_protocol(source_path)
        cycle_index, cell_id = self._extract_cycle_and_cell(source_path)
        metadata: dict[MappingSpec, object] = {
            BaseColumns.proto: protocol_id,
            BaseColumns.cell_id: cell_id,
            BaseColumns.cidx: cycle_index,
        }
        if soc_pct is not None:
            metadata[BaseColumns.soc_pct] = soc_pct
        return metadata

    def _infer_protocol(self, source_path: str) -> tuple[DatasetProtocolId, float | None]:
        if self.EIS_FILE_INDICATOR in source_path:
            return DatasetProtocolId.eis, self._infer_eis_soc_pct(source_path)
        if self.HPPC_FILE_INDICATOR in source_path:
            return DatasetProtocolId.hppc, None
        if self.DIAG_FILE_INDICATOR in source_path:
            return DatasetProtocolId.rpt, None
        if self.CYCLING_FILE_INDICATOR in source_path:
            return DatasetProtocolId.cycling, None
        raise ValueError(f"Could not infer protocol from source path: {source_path}")

    @staticmethod
    def _infer_eis_soc_pct(source_path: str) -> float:
        match = re.search(r"(?:SOC|EIS)(\d+(?:\.\d+)?)", source_path)
        if match is None:
            raise ValueError(f"Could not infer EIS SOC metadata from source path: {source_path}")
        return float(match.group(1))

    @staticmethod
    def _extract_cycle_and_cell(source_path: str) -> tuple[int, str]:
        path = Path(source_path)
        parts = path.parts
        stem = path.stem
        for idx, part in enumerate(parts):
            cycling_match = re.fullmatch(r"Cycling_(\d+)", part)
            if cycling_match is not None:
                if idx + 1 >= len(parts) or re.fullmatch(r"[A-Za-z]\d+", parts[idx + 1]) is None:
                    raise ValueError(f"Invalid cycling cell id in path: {source_path}")
                return int(cycling_match.group(1)), parts[idx + 1]
            diag_match = re.fullmatch(r"Diag_(\d+)", part)
            if diag_match is not None:
                return int(diag_match.group(1)) - 1, _extract_cell_id(stem)

        cycle_match = re.search(r"(?:Cycling|Diag|HPPC)_(\d+)", stem)
        if cycle_match is not None:
            return int(cycle_match.group(1)), _extract_cell_id(stem)
        raise ValueError(f"Could not infer cycle index and cell id from source path: {source_path}")

    def _source_sort_key(self, source_path: str) -> tuple[int, int, str, float, int, int, str]:
        protocol_id, soc_pct = self._infer_protocol(source_path)
        cycle_index, cell_id = self._extract_cycle_and_cell(source_path)
        part_num, channel_num = self._extract_part_and_channel(source_path)
        return (
            self.PROTOCOL_SORT_ORDER.get(protocol_id, 99),
            cycle_index,
            cell_id,
            -1.0 if soc_pct is None else soc_pct,
            part_num,
            channel_num,
            source_path,
        )

    @classmethod
    def _extract_part_and_channel(cls, source_path: str) -> tuple[int, int]:
        stem = Path(source_path).stem
        part_num = 0
        channel_num = 0
        if cls.PART_INDICATOR in stem:
            parts = stem.split(cls.PART_INDICATOR)
            if len(parts) > 1:
                part_section = parts[1].split("_")[0]
                if part_section.isdigit():
                    part_num = int(part_section)

        for delimiter in (".", "_"):
            if delimiter not in stem:
                continue
            last_part = stem.split(delimiter)[-1]
            if last_part.isdigit():
                channel_num = int(last_part)
                break
        return part_num, channel_num


def _extract_cell_id(value: str) -> str:
    channel_prefix = value.split("_Channel_", maxsplit=1)[0]
    matches = [token for token in channel_prefix.split("_") if re.fullmatch(r"[A-Za-z]\d+", token)]
    if not matches:
        raise ValueError(f"Could not infer cell id from value: {value}")
    return matches[-1]
