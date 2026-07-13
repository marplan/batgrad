from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.contracts.row_ids import MANIFEST_ROW_ID_COLUMN, ML_INDEX_ROW_ID_COLUMN
from batgrad.ml.config import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    LoaderTrainConfig,
    LoggingConfig,
    MaskedSuffixConfig,
    OptimizerConfig,
    RunConfig,
    ScalingRuleConfig,
    SchedulerConfig,
    TrainConfig,
    ValidationConfig,
    ValidationGroupConfig,
    ValidationMaskedSuffixConfig,
    ValidationSplitConfig,
)
from batgrad.ml.data.batch import Batch, BatchState
from batgrad.ml.data.index import MlDatasetIndex
from batgrad.ml.nn import LayerConfig, MambaCarryState, SequenceMixerConfig

INPUT_COLUMNS = ("time", "current", "voltage", "temperature")
TARGET_COLUMNS = ("voltage", "temperature")
FEEDBACK_COLUMNS = TARGET_COLUMNS
MASKED_CHANNELS = TARGET_COLUMNS
TINY_MANIFEST_PATH = "type=synthetic/dataset=tiny-ml/source=normalized/manifest.parquet"
TINY_GIT_COMMIT = "abcdef0"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "/tests/ml/" in str(item.path):
            item.add_marker(pytest.mark.cpu_only)


class InMemoryMlStore:
    def __init__(
        self,
        tables: dict[str, pl.DataFrame] | None = None,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.root = "memory://ml"
        self.tables = dict(tables or {})
        self.files = dict(files or {})
        self.slices: list[tuple[str, tuple[tuple[int, int], ...], tuple[str, ...] | None]] = []

    def list_files(self, location: str | Path | None = None, pattern: str = "*") -> tuple[str, ...]:
        del location, pattern
        return tuple(sorted(self.tables))

    def local_file(self, location: str | Path):
        path = str(location)
        if path in self.files:
            return nullcontext(BytesIO(self.files[path]))
        return nullcontext(Path(str(location)))

    def scan_table(
        self,
        location: str | Path | tuple[str | Path, ...],
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
        limit: int | None = None,
    ) -> pl.LazyFrame:
        if isinstance(location, tuple):
            frame = pl.concat([self._table(path) for path in location], how="vertical")
        else:
            frame = self._table(location)
        if columns is not None:
            frame = frame.select(columns)
        if filters is not None:
            frame = frame.filter(filters)
        if limit is not None:
            frame = frame.limit(limit)
        return frame.lazy()

    def iter_table_chunks(
        self,
        location: str | Path | tuple[str | Path, ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
        filters: pl.Expr | None = None,
    ):
        frame = self.scan_table(location, columns=columns, filters=filters).collect()
        for offset in range(0, frame.height, chunk_rows):
            yield frame.slice(offset, chunk_rows)

    def iter_table_slices(
        self,
        location: str | Path,
        slices: tuple[tuple[int, int], ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
    ):
        self.slices.append((str(location), slices, columns))
        table = self._table(location)
        if columns is not None:
            table = table.select(columns)
        for offset, rows in slices:
            sliced = table.slice(offset, rows)
            for chunk_offset in range(0, sliced.height, chunk_rows):
                yield sliced.slice(chunk_offset, chunk_rows)

    def table_size_bytes(self, location: str | Path) -> int | None:
        return self._table(location).estimated_size()

    def _table(self, location: str | Path) -> pl.DataFrame:
        path = str(location)
        if path not in self.tables:
            raise FileNotFoundError(path)
        return self.tables[path]


class RecordingModel(torch.nn.Module):
    def __init__(
        self,
        target_count: int = 2,
        num_bins: int = 5,
        *,
        position_bins: bool = False,
    ) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.target_count = target_count
        self.num_bins = num_bins
        self.position_bins = position_bins
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        states: dict[str, MambaCarryState] | None = None,
        return_states: bool = False,
    ):
        self.calls.append(
            {
                "x": x.detach().clone(),
                "mask": None if mask is None else mask.detach().clone(),
                "states": states,
                "return_states": return_states,
            }
        )
        batch, seq_len = int(x.shape[0]), int(x.shape[1])
        logits = torch.zeros(
            (batch, seq_len, self.target_count, self.num_bins), dtype=x.dtype, device=x.device
        )
        logits = logits + self.bias
        if self.position_bins:
            position_bins = (
                torch.arange(seq_len, device=x.device).view(1, seq_len, 1, 1) % self.num_bins
            )
            logits.scatter_(-1, position_bins.expand(batch, seq_len, self.target_count, 1), 2.0)
        else:
            # Prefer the high bin so decoded feedback changes from the original scalar inputs.
            logits[..., -1] = logits[..., -1] + 2.0
        if not return_states:
            return logits
        return logits, {"layer": fake_mamba_state(float(len(self.calls)), device=x.device)}


class StateProbeModel(torch.nn.Module):
    """Deterministic model double that makes state hand-offs observable."""

    def __init__(
        self,
        target_count: int = 2,
        num_bins: int = 5,
        *,
        advance_by_sequence_length: bool = False,
    ) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.target_count = target_count
        self.num_bins = num_bins
        self.advance_by_sequence_length = advance_by_sequence_length
        self.calls: list[dict[str, Any]] = []

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        states: dict[str, MambaCarryState] | None = None,
        return_states: bool = False,
    ):
        received_state_value = (
            None if states is None else float(states["layer"].angle_state[0, 0].detach().cpu())
        )
        self.calls.append(
            {
                "x": x.detach().clone(),
                "mask": None if mask is None else mask.detach().clone(),
                "states": states,
                "received_state_value": received_state_value,
                "return_states": return_states,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        batch, seq_len = int(x.shape[0]), int(x.shape[1])
        logits = torch.zeros(
            (batch, seq_len, self.target_count, self.num_bins), dtype=x.dtype, device=x.device
        )
        logits = logits + self.bias
        logits[..., -1] = logits[..., -1] + 2.0
        if not return_states:
            return logits
        state_increment = float(seq_len) if self.advance_by_sequence_length else 1.0
        state_value = (received_state_value or 0.0) + state_increment
        return logits, {"layer": fake_mamba_state(state_value, device=x.device)}


class MetricLogger:
    def __init__(self) -> None:
        self.metrics: list[tuple[int, dict[str, object]]] = []
        self.payloads: list[tuple[int, str, object]] = []

    def log_metrics(self, step: int, metrics: dict[str, object], **_: object) -> None:
        self.metrics.append((step, dict(metrics)))

    def log_payload(self, step: int, name: str, payload: object) -> None:
        self.payloads.append((step, name, payload))

    def run_name(self) -> str | None:
        return None

    def run_id(self) -> str | None:
        return None

    def finish(self) -> None:
        return


@pytest.fixture
def ml_store() -> InMemoryMlStore:
    return make_store()


@pytest.fixture
def ml_index() -> MlDatasetIndex:
    return make_index()


@pytest.fixture
def tiny_config() -> ExperimentConfig:
    return make_config()


def make_store(rows: int = 80) -> InMemoryMlStore:
    tables: dict[str, pl.DataFrame] = {}
    for cell_idx, cell in enumerate(("cell-a", "cell-b")):
        for cycle in (1, 2):
            for protocol in (DatasetProtocolId.cycling, DatasetProtocolId.hppc):
                path = shard_path(cell, cycle, protocol)
                tables[path] = series_frame(rows, offset=1000 * cell_idx + 100 * cycle)
    return InMemoryMlStore(tables)


def make_memory_manifest_store(rows: int = 44) -> InMemoryMlStore:
    tables: dict[str, pl.DataFrame] = {}
    records: list[dict[str, object]] = []
    for cell_idx, cell in enumerate(("cell-a", "cell-b")):
        for cycle in (1, 2):
            for protocol in (DatasetProtocolId.cycling, DatasetProtocolId.hppc):
                shard = shard_path(cell, cycle, protocol, dataset="tiny-ml")
                tables[shard] = series_frame(rows, offset=1000 * cell_idx + 100 * cycle)
                records.append(
                    {
                        BaseColumns.set_id: "tiny-ml",
                        BaseColumns.cell_id: cell,
                        BaseColumns.cidx: cycle,
                        BaseColumns.proto: str(protocol),
                        BaseColumns.row_n: rows,
                        BaseColumns.norm_stats: [
                            {"column": column, "min": 0.0, "max": 2000.0}
                            for column in INPUT_COLUMNS
                        ],
                        BaseColumns.norm_segs: [_segment(shard, rows)],
                    }
                )
    tables[TINY_MANIFEST_PATH] = pl.DataFrame(records)
    return InMemoryMlStore(
        tables,
        {
            TINY_MANIFEST_PATH: manifest_footer_bytes(
                {
                    str(BaseColumns.git_commit): TINY_GIT_COMMIT,
                    str(BaseColumns.git_status): str(BaseColumns.git_status.values.clean),
                }
            )
        },
    )


def manifest_footer_bytes(metadata: dict[str, str]) -> bytes:
    buffer = BytesIO()
    table = pa.table({"footer": [0]}).replace_schema_metadata(
        {key.encode(): value.encode() for key, value in metadata.items()}
    )
    pq.write_table(table, buffer)
    return buffer.getvalue()


def make_index(rows: int = 80, *, split_cell_b: bool = True) -> MlDatasetIndex:
    records: list[dict[str, object]] = []
    row_id = 0
    for cell in ("cell-a", "cell-b"):
        for cycle in (1, 2):
            for protocol in (DatasetProtocolId.cycling, DatasetProtocolId.hppc):
                split = (
                    BaseColumns.split.values.val
                    if split_cell_b and cell == "cell-b" and cycle == 2
                    else BaseColumns.split.values.train
                )
                records.append(
                    {
                        ML_INDEX_ROW_ID_COLUMN: row_id,
                        MANIFEST_ROW_ID_COLUMN: row_id,
                        BaseColumns.set_id: "synthetic-ml",
                        BaseColumns.cell_id: cell,
                        BaseColumns.cidx: cycle,
                        BaseColumns.proto: str(protocol),
                        BaseColumns.split: split,
                        BaseColumns.manifest: "memory-manifest.parquet",
                        BaseColumns.row_n: rows,
                        BaseColumns.norm_segs: [_segment(shard_path(cell, cycle, protocol), rows)],
                    }
                )
                row_id += 1
    return MlDatasetIndex(pl.DataFrame(records))


def make_config(
    *,
    batch_size: int = 3,
    seq_len: int = 10,
    suffix_steps: int = 3,
    roll_forward_steps: int = 0,
    loss_on_masked_only: bool = True,
    carry_mamba_state: bool = True,
    feedback_mode: str = "probabilities",
    validation_masked_suffix: ValidationMaskedSuffixConfig | None = None,
) -> ExperimentConfig:
    data = DataConfig(
        manifest_paths={"memory-manifest.parquet": "abcdef0"},
        protocols=("cycling", "HPPC"),
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        feedback_columns=FEEDBACK_COLUMNS,
        scaling=tuple(
            ScalingRuleConfig(column=column, input_min=-10.0, input_max=10.0)
            for column in INPUT_COLUMNS
        ),
    )
    model = SequenceMixerConfig(
        d_model=8,
        n_heads=2,
        num_bins=5,
        feedback_mode=feedback_mode,  # type: ignore[arg-type]
        layers=(LayerConfig(kind="ffn"),),
        head_layers=(LayerConfig(kind="ffn"),),
    )
    return ExperimentConfig(
        data=data,
        loader=LoaderTrainConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            strategy="shuffled_protocol_groups",
            stateful_n_windows=2,
            cross_protocol_state_carry="chain",
            data_access="windowed",
            num_workers=0,
        ),
        model=model,
        train=TrainConfig(
            epochs=1.0,
            log_every_steps=1,
            validate_every_steps=1,
            max_steps=2,
            masked_suffix=MaskedSuffixConfig(
                enabled=True,
                channels=MASKED_CHANNELS,
                suffix_steps=suffix_steps,
                loss_on_masked_only=loss_on_masked_only,
                carry_mamba_state=carry_mamba_state,
                detach_between_windows=True,
                roll_forward_steps=roll_forward_steps,
            ),
        ),
        validation=ValidationConfig(
            split=ValidationSplitConfig(
                strategy="provide",
                group_by=(
                    BaseColumns.set_id,
                    BaseColumns.cell_id,
                    BaseColumns.cidx,
                    BaseColumns.proto,
                ),
                groups=(
                    ValidationGroupConfig(
                        match={
                            BaseColumns.set_id: "synthetic-ml",
                            BaseColumns.cell_id: "cell-b",
                            BaseColumns.cidx: 2,
                            BaseColumns.proto: str(DatasetProtocolId.cycling),
                        },
                        rollout_start_offsets=(12,),
                    ),
                ),
            ),
            max_tf_batches=1,
            rollout_steps=4,
            log_rollout_plots=False,
            masked_suffix=validation_masked_suffix or ValidationMaskedSuffixConfig(),
        ),
        optim=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(kind="none"),
        run=RunConfig(device="cpu", use_amp=False, compile_model=False, output_dir=None),
        logging=LoggingConfig(backend="stdout"),
        checkpoint=CheckpointConfig(),
    )


def make_batch(
    config: ExperimentConfig | None = None,
    *,
    seq_len: int | None = None,
    stateful_group_idx: int | None = None,
    stateful_step_idx: int | None = None,
    stateful_steps: int | None = None,
    all_valid: bool = True,
) -> Batch:
    cfg = config or make_config()
    batch_size = cfg.loader.batch_size
    length = seq_len or cfg.loader.seq_len + cfg.train.masked_suffix.roll_forward_steps
    base = torch.arange(batch_size * length, dtype=torch.float32).reshape(batch_size, length)
    inputs = torch.stack((base / 100.0, base / 50.0, base / 25.0, base / 10.0), dim=-1)
    targets = inputs[..., 2:4].clone()
    mask = torch.ones((batch_size, length), dtype=torch.bool)
    if not all_valid:
        mask[-1, -2:] = False
    return Batch(
        inputs=inputs,
        targets=targets,
        mask=mask,
        all_valid=bool(mask.all().item()),
        state=BatchState(
            split=BaseColumns.split.values.train,
            batch_idx=0,
            protocols=(DatasetProtocolId.cycling,) * batch_size,
            manifest_paths=("memory-manifest.parquet",) * batch_size,
            manifest_row_ids=tuple(range(batch_size)),
            group_keys=tuple(
                ("synthetic-ml", f"cell-{idx}", 1, str(DatasetProtocolId.cycling))
                for idx in range(batch_size)
            ),
            alignment_keys=tuple(("synthetic-ml", f"cell-{idx}", 1) for idx in range(batch_size)),
            segments=(),
            window_offsets=tuple(range(batch_size)),
            stateful_group_idx=stateful_group_idx,
            stateful_step_idx=stateful_step_idx,
            stateful_steps=stateful_steps,
        ),
    )


def fake_mamba_state(value: float = 1.0, *, device: torch.device | str = "cpu") -> MambaCarryState:
    tensor = torch.full((2, 2), value, dtype=torch.float32, device=device, requires_grad=True)
    return MambaCarryState(
        angle_state=tensor,
        ssm_state=tensor + 1.0,
        k_state=tensor + 2.0,
        v_state=tensor + 3.0,
    )


def config_with(config: ExperimentConfig, **kwargs: object) -> ExperimentConfig:
    return replace(config, **kwargs)


def series_frame(rows: int, *, offset: int = 0) -> pl.DataFrame:
    values = [float(offset + idx) for idx in range(rows)]
    return pl.DataFrame(
        {
            "time": values,
            "current": [value / 10.0 for value in values],
            "voltage": [3.0 + (idx % 10) / 100.0 for idx in range(rows)],
            "temperature": [20.0 + (idx % 5) for idx in range(rows)],
        }
    )


def shard_path(
    cell: str,
    cycle: int,
    protocol: DatasetProtocolId,
    *,
    dataset: str = "synthetic-ml",
) -> str:
    return (
        f"type=synthetic/dataset={dataset}/source=normalized/"
        f"cell={cell}/cycle={cycle}/protocol={protocol!s}.parquet"
    )


def _segment(path: str, rows: int) -> dict[str, object]:
    return {
        str(BaseColumns.path): path,
        str(BaseColumns.row0): 0,
        str(BaseColumns.row_n): rows,
    }
