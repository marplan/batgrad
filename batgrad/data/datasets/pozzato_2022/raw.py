from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import fastexcel
import polars as pl

from batgrad import _loggers
from batgrad.contracts.columns import BaseColumns, ColumnSpec, MetadataColumns, collect_column_specs
from batgrad.contracts.values import BaseValues
from batgrad.data.datasets.pozzato_2022.specs import DATASET_SPEC
from batgrad.data.processing.config import FailureMode
from batgrad.data.processing.raw import (
    RawBatch,
    RawIngestIssue,
    RawTask,
    RawTaskStats,
    is_excluded_raw_file,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batgrad.data.datasets.specs import DatasetSpec
    from batgrad.storage.store import DataStore

logger = _loggers.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SchemaApplyResult:
    data: pl.DataFrame
    stats: RawTaskStats


@dataclass(frozen=True, slots=True)
class Pozzato2022RawAdapter:
    spec: DatasetSpec = DATASET_SPEC

    EIS_SHEET_PATTERN: ClassVar[str] = "ACIM"
    TIMESERIES_SHEET_PATTERN: ClassVar[str] = "channel"
    EXCLUDED_SHEET_PATTERN: ClassVar[str] = "chart"

    EIS_FILE_INDICATOR: ClassVar[str] = "EIS"
    HPPC_FILE_INDICATOR: ClassVar[str] = "HPPC"
    DIAG_FILE_INDICATOR: ClassVar[str] = "Diag"
    CYCLING_FILE_INDICATOR: ClassVar[str] = "Cycling"

    CYCLING_FILE_PREFIX: ClassVar[str] = "Cycling_"
    DIAG_FILE_PREFIX: ClassVar[str] = "Diag_"
    PART_INDICATOR: ClassVar[str] = "Part"
    PROTOCOL_SORT_ORDER: ClassVar[dict[str, int]] = {
        BaseValues.cycling_protocol: 0,
        BaseValues.hppc_protocol: 1,
        BaseValues.rpt_protocol: 2,
        BaseValues.eis_protocol: 3,
    }

    NOMINAL_CAPACITY_AH: ClassVar[float] = 5.0

    def plan_raw_tasks(self, input_store: DataStore) -> tuple[RawTask, ...]:
        raw_spec = self.spec.raw
        if raw_spec is None:
            raise ValueError(f"Dataset {self.spec.dataset_id!r} does not support raw ingestion")

        raw_root = self.spec.location.source_root(raw_spec.input_source)
        paths: list[str] = []

        for suffix in raw_spec.file_suffixes:
            for path in input_store.list_files(raw_root, pattern=f"*{suffix}"):
                if is_excluded_raw_file(path, raw_spec):
                    continue
                paths.append(path)

        return tuple(
            RawTask(
                task_id=path,
                source_paths=(path,),
            )
            for path in sorted(paths, key=self._source_sort_key)
        )

    def load_raw_task(
        self,
        task: RawTask,
        input_store: DataStore,
        failure_mode: FailureMode,
    ) -> Iterator[RawBatch]:
        source_path = task.source_paths[0]
        raw_spec = self.spec.raw
        if raw_spec is None:
            raise ValueError(f"Dataset {self.spec.dataset_id!r} does not support raw ingestion")
        metadata = self._extract_metadata(source_path)
        protocol_schema = raw_spec.protocol_schema(metadata[MetadataColumns.protocol])

        with input_store.local_file(source_path) as path:
            excel = fastexcel.read_excel(path)
            sheet_name = self._select_sheet(source_path, tuple(excel.sheet_names))
            sheet = excel.load_sheet(idx_or_name=sheet_name, dtypes="string")

        schema_result = self._apply_schema(
            sheet.to_polars(),
            source_path,
            protocol_schema.columns,
            protocol_schema.dropped_columns,
            failure_mode,
        )
        data = schema_result.data

        if metadata[MetadataColumns.protocol] != BaseValues.eis_protocol:
            current_col = collect_column_specs(self.spec.cols)["current"]
            if current_col in data.columns:
                data = data.with_columns((pl.col(current_col) * -1).alias(current_col))

        yield RawBatch(
            data=data,
            stream_id=source_path,
            source_paths=(source_path,),
            metadata=metadata,
            stats=schema_result.stats,
        )

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
            raise ValueError(
                f"Expected exactly one data sheet in {source_path}, got {matches}",
            )

        return matches[0]

    def _apply_schema(
        self,
        data: pl.DataFrame,
        source_path: str,
        declared_columns: tuple[ColumnSpec, ...],
        dropped_columns: tuple[ColumnSpec, ...],
        failure_mode: FailureMode,
    ) -> SchemaApplyResult:
        select_exprs: list[pl.Expr] = []
        output_counts: dict[str, int] = {}
        unknown_columns: list[str] = []
        declared_drop_columns: list[str] = []
        duplicate_issues: list[RawIngestIssue] = []

        for source_col in data.columns:
            spec, is_declared_drop = self._resolve_source_column(
                source_col,
                source_path,
                declared_columns,
                dropped_columns,
            )
            if spec is None:
                if is_declared_drop:
                    declared_drop_columns.append(source_col)
                else:
                    unknown_columns.append(source_col)
                continue

            output_col = self._duplicate_output_column(spec, output_counts)
            if output_col != spec:
                duplicate_issues.append(
                    self._duplicate_column_issue(source_path, source_col, spec, output_col),
                )
            expr = pl.col(source_col)
            if spec.dtype is not None:
                expr = expr.cast(spec.dtype, strict=False)
            select_exprs.append(expr.alias(output_col))

        issues: list[RawIngestIssue] = []
        if declared_drop_columns:
            issues.append(
                RawIngestIssue(
                    kind="dropped_declared_columns",
                    source_paths=(source_path,),
                    message=(
                        f"Raw file {source_path} has columns declared for dropping: "
                        f"{declared_drop_columns}"
                    ),
                    columns=tuple(declared_drop_columns),
                ),
            )
        if unknown_columns:
            message = (
                f"Raw file {source_path} has columns outside the protocol schema: {unknown_columns}"
            )
            issues.append(
                RawIngestIssue(
                    kind="dropped_unknown_columns",
                    source_paths=(source_path,),
                    message=f"{message}; dropping columns",
                    columns=tuple(unknown_columns),
                ),
            )
            if failure_mode == FailureMode.STRICT:
                for issue in issues:
                    logger.warning(
                        "raw ingest issue kind=%s source_paths=%s columns=%s message=%s",
                        issue.kind,
                        issue.source_paths,
                        issue.columns,
                        issue.message,
                    )
                raise ValueError(message)

        issues.extend(duplicate_issues)
        stats = RawTaskStats(
            dropped_unknown_columns=len(unknown_columns),
            dropped_declared_columns=len(declared_drop_columns),
            duplicate_columns=len(duplicate_issues),
            issues=tuple(issues),
        )
        return SchemaApplyResult(data=data.select(select_exprs), stats=stats)

    @staticmethod
    def _resolve_source_column(
        source_col: str,
        source_path: str,
        declared_columns: tuple[ColumnSpec, ...],
        dropped_columns: tuple[ColumnSpec, ...],
    ) -> tuple[ColumnSpec | None, bool]:
        matches = [spec for spec in declared_columns if spec.has_match(source_col)]
        if not matches:
            if any(spec.has_match(source_col) for spec in dropped_columns):
                return None, True
            return None, False
        if len(matches) > 1:
            logger.warning(
                "Raw file %s source column %r matched multiple declared columns: %s",
                source_path,
                source_col,
                matches,
            )
            raise ValueError(
                f"Source column {source_col!r} matched multiple declared columns "
                f"in {source_path}: {matches}",
            )
        return matches[0], False

    @staticmethod
    def _duplicate_column_issue(
        source_path: str,
        source_col: str,
        spec: ColumnSpec,
        output_col: str,
    ) -> RawIngestIssue:
        return RawIngestIssue(
            kind="duplicate_columns",
            source_paths=(source_path,),
            message=(
                f"Raw file {source_path} maps duplicate source column {source_col!r} "
                f"to {spec!r}; using output column {output_col!r}"
            ),
            columns=(source_col, output_col),
        )

    @staticmethod
    def _duplicate_output_column(spec: ColumnSpec, output_counts: dict[str, int]) -> str:
        count = output_counts.get(spec, 0)
        output_counts[spec] = count + 1
        if count == 0:
            return spec
        return f"{spec} {count}"

    def _extract_metadata(self, source_path: str) -> dict[ColumnSpec, object]:
        protocol, domain_id, soc_pct = self._infer_protocol(source_path)
        cycle_index, cell_id = self._extract_cycle_and_cell(source_path)
        metadata: dict[ColumnSpec, object] = {
            MetadataColumns.protocol: protocol,
            MetadataColumns.domain_id: domain_id,
            BaseColumns.cell_id: cell_id,
            BaseColumns.cycle_index: cycle_index,
        }
        if soc_pct is not None:
            metadata[MetadataColumns.soc_pct] = soc_pct
        return metadata

    def _infer_protocol(self, source_path: str) -> tuple[str, str, float | None]:
        if self.EIS_FILE_INDICATOR in source_path:
            return (
                BaseValues.eis_protocol,
                BaseValues.freq_domain,
                self._infer_eis_soc_pct(source_path),
            )
        if self.HPPC_FILE_INDICATOR in source_path:
            return BaseValues.hppc_protocol, BaseValues.time_domain, None
        if self.DIAG_FILE_INDICATOR in source_path:
            return BaseValues.rpt_protocol, BaseValues.time_domain, None
        if self.CYCLING_FILE_INDICATOR in source_path:
            return BaseValues.cycling_protocol, BaseValues.time_domain, None
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
        protocol, _domain_id, soc_pct = self._infer_protocol(source_path)
        cycle_index, cell_id = self._extract_cycle_and_cell(source_path)
        part_num, channel_num = self._extract_part_and_channel(source_path)
        return (
            self.PROTOCOL_SORT_ORDER.get(protocol, 99),
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
