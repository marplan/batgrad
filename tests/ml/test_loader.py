from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.data.config import LoaderConfig, ScalingRule, ValidationConfig, WindowConfig
from batgrad.ml.data.index import MANIFEST_ROW_ID_COLUMN, MlDatasetIndex
from batgrad.ml.data.loader import (
    MlDataIterable,
    create_dataloader,
    dataloader_for_split,
)
from batgrad.ml.data.materialization import materialize_window_ref
from batgrad.ml.data.planning import (
    StreamPlan,
    WindowRef,
    _build_phase_offsets,
    build_batch_plans,
    count_batch_plans,
)
from batgrad.ml.data.scaling import (
    inverse_scale_tensor,
    minmax_scaling,
    scale_data,
)
from batgrad.storage.local import LocalDataProcessingStore
from notebooks.ml_helpers import make_batch_preview_submission, selected_preview_scaling


class RecordingStore(LocalDataProcessingStore):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root, create=True)
        self.slices: list[tuple[str, tuple[tuple[int, int], ...], tuple[str, ...] | None]] = []

    def iter_table_slices(
        self,
        location: str | Path,
        slices: tuple[tuple[int, int], ...],
        chunk_rows: int,
        columns: tuple[str, ...] | None = None,
    ):
        self.slices.append((str(location), slices, columns))
        yield from super().iter_table_slices(location, slices, chunk_rows, columns)


def test_shuffled_plans_do_not_overwrite_duplicate_group_keys() -> None:
    index = MlDatasetIndex(
        pl.DataFrame(
            {
                BaseColumns.set_id: ["set-a", "set-a"],
                BaseColumns.cell_id: ["cell-a", "cell-a"],
                BaseColumns.cidx: [1, 1],
                BaseColumns.proto: [str(DatasetProtocolId.cycling)] * 2,
                BaseColumns.split: [BaseColumns.split.values.train] * 2,
                BaseColumns.manifest: ["manifest-a.parquet", "manifest-b.parquet"],
                MANIFEST_ROW_ID_COLUMN: [0, 0],
                BaseColumns.row_n: [10, 10],
                BaseColumns.norm_segs: [
                    [_segment("a.parquet", 0, 10)],
                    [_segment("b.parquet", 0, 10)],
                ],
            }
        )
    )
    config = LoaderConfig(
        default_window=WindowConfig(batch_size=2, seq_len=2),
        stateful_n_windows=1,
    )

    plans = build_batch_plans(index, DatasetProtocolId.cycling, config, epoch_idx=0)

    manifest_paths = {ref.manifest_path for plan in plans for ref in plan.refs}
    assert manifest_paths == {"manifest-a.parquet", "manifest-b.parquet"}
    assert len(plans) == 4
    assert all(len(plan.refs) == 2 for plan in plans)


def test_count_batch_plans_matches_shuffled_plans() -> None:
    index = MlDatasetIndex(
        pl.DataFrame(
            {
                BaseColumns.set_id: ["set-a", "set-a"],
                BaseColumns.cell_id: ["cell-a", "cell-b"],
                BaseColumns.cidx: [1, 1],
                BaseColumns.proto: [str(DatasetProtocolId.cycling)] * 2,
                BaseColumns.split: [BaseColumns.split.values.train] * 2,
                BaseColumns.manifest: ["manifest-a.parquet", "manifest-b.parquet"],
                MANIFEST_ROW_ID_COLUMN: [0, 1],
                BaseColumns.row_n: [25, 33],
                BaseColumns.norm_segs: [
                    [_segment("a.parquet", 0, 25)],
                    [_segment("b.parquet", 0, 33)],
                ],
            }
        )
    )
    config = LoaderConfig(
        default_window=WindowConfig(batch_size=3, seq_len=4),
        stateful_n_windows=2,
    )

    plans = build_batch_plans(index, DatasetProtocolId.cycling, config, epoch_idx=1)

    assert count_batch_plans(index, DatasetProtocolId.cycling, config, epoch_idx=1) == len(plans)


def test_dataset_steps_per_epoch_uses_planned_batches(tmp_path: Path) -> None:
    index = MlDatasetIndex(
        pl.DataFrame(
            {
                BaseColumns.set_id: ["set-a"],
                BaseColumns.cell_id: ["cell-a"],
                BaseColumns.cidx: [1],
                BaseColumns.proto: [str(DatasetProtocolId.cycling)],
                BaseColumns.split: [BaseColumns.split.values.train],
                BaseColumns.manifest: ["manifest.parquet"],
                MANIFEST_ROW_ID_COLUMN: [0],
                BaseColumns.row_n: [100],
                BaseColumns.norm_segs: [[_segment("a.parquet", 0, 100)]],
            }
        )
    )
    config = LoaderConfig(
        default_window=WindowConfig(batch_size=2, seq_len=5),
        stateful_n_windows=1,
    )
    dataset = MlDataIterable(
        store=RecordingStore(tmp_path / "store"),
        index=index,
        input_columns=(BaseColumns.time,),
        target_columns=(BaseColumns.volt,),
        scaling=(),
        config=config,
        active_protocol=DatasetProtocolId.cycling,
    )

    assert dataset.steps_per_epoch(0) == count_batch_plans(
        dataset.index,
        DatasetProtocolId.cycling,
        config,
        epoch_idx=0,
        stream_plans=dataset.stream_plans,
    )


def test_phase_offsets_keep_epoch_shift_even_when_count_changes() -> None:
    assert _build_phase_offsets(stream_len=1000, step=128, phase=120) == (
        120,
        248,
        376,
        504,
        632,
        760,
    )


def test_materialize_window_ref_reads_only_requested_window(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(
        pl.DataFrame(
            {
                BaseColumns.time: list(range(20)),
                BaseColumns.curr: [float(value) for value in range(20)],
                BaseColumns.volt: [100.0 + value for value in range(20)],
            }
        ),
        "data.parquet",
        row_group_size=5,
    )
    config = LoaderConfig(
        strategy="sequential",
        default_window=WindowConfig(batch_size=2, seq_len=3),
    )
    stream = StreamPlan(
        protocol=DatasetProtocolId.cycling,
        split=BaseColumns.split.values.train,
        manifest_path="manifest.parquet",
        manifest_row_id=0,
        stream_identity=(
            ("set-a", "cell-a", 1, DatasetProtocolId.cycling),
            DatasetProtocolId.cycling,
            "manifest.parquet",
            0,
        ),
        group_key=("set-a", "cell-a", 1, DatasetProtocolId.cycling),
        alignment_key=("set-a", "cell-a", 1),
        segments=({"path": "data.parquet", "row_start": 0, "row_count": 20},),
        row_count=20,
        phase_start=0,
        phase_stride=1,
    )
    ref = WindowRef(stream=stream, offset=7)

    batch = materialize_window_ref(
        store,
        ref,
        (BaseColumns.time, BaseColumns.curr),
        (BaseColumns.volt,),
        (),
        config,
        batch_idx=0,
    )

    assert store.slices == [
        ("data.parquet", ((7, 7),), (BaseColumns.time, BaseColumns.curr, BaseColumns.volt))
    ]
    assert tuple(batch.active.inputs.shape) == (2, 3, 2)
    assert tuple(batch.active.targets.shape) == (2, 3, 1)
    assert batch.active.all_valid is True
    assert batch.active.inputs[0, 0, 0].item() == 7.0
    assert batch.active.targets[0, 0, 0].item() == 108.0


def test_materialize_window_ref_marks_padded_batch_not_all_valid(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(
        pl.DataFrame(
            {
                BaseColumns.time: [float(value) for value in range(20)],
                BaseColumns.volt: [100.0 + value for value in range(20)],
            }
        ),
        "data.parquet",
        row_group_size=5,
    )
    stream = StreamPlan(
        protocol=DatasetProtocolId.cycling,
        split=BaseColumns.split.values.train,
        manifest_path="manifest.parquet",
        manifest_row_id=0,
        stream_identity=(
            ("set-a", "cell-a", 1, DatasetProtocolId.cycling),
            DatasetProtocolId.cycling,
            "manifest.parquet",
            0,
        ),
        group_key=("set-a", "cell-a", 1, DatasetProtocolId.cycling),
        alignment_key=("set-a", "cell-a", 1),
        segments=(_segment("data.parquet", 0, 20),),
        row_count=20,
        phase_start=0,
        phase_stride=1,
    )

    batch = materialize_window_ref(
        store,
        WindowRef(stream=stream, offset=18),
        (BaseColumns.time,),
        (BaseColumns.volt,),
        (),
        LoaderConfig(strategy="sequential", default_window=WindowConfig(batch_size=2, seq_len=3)),
        batch_idx=0,
    )

    assert batch.active.all_valid is False
    assert batch.active.mask.tolist() == [[True, False, False], [False, False, False]]


def test_full_in_mem_loader_matches_windowed_loader(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(
        pl.DataFrame(
            {
                BaseColumns.time: list(range(30)),
                BaseColumns.curr: [float(value) for value in range(30)],
                BaseColumns.volt: [100.0 + value for value in range(30)],
            }
        ),
        "data.parquet",
        row_group_size=5,
    )
    index = _index_for_store("data.parquet", rows=30)
    window = WindowConfig(batch_size=2, seq_len=3)
    common = {
        "store": store,
        "index": index,
        "input_columns": (BaseColumns.time, BaseColumns.curr),
        "target_columns": (BaseColumns.volt,),
        "scaling": (),
        "active_protocol": DatasetProtocolId.cycling,
    }
    windowed = next(iter(MlDataIterable(config=LoaderConfig(default_window=window), **common)))
    full_in_mem = next(
        iter(
            MlDataIterable(
                config=LoaderConfig(default_window=window, data_access="full_in_mem"),
                **common,
            )
        )
    )

    assert full_in_mem.active.inputs.equal(windowed.active.inputs)
    assert full_in_mem.active.targets.equal(windowed.active.targets)
    assert full_in_mem.active.mask.equal(windowed.active.mask)
    assert store.slices


def test_full_in_mem_rejects_workers() -> None:
    with pytest.raises(ValueError, match="requires num_workers=0"):
        LoaderConfig(data_access="full_in_mem", num_workers=1)


def test_dataloader_for_split_uses_full_index(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(_series_frame(20), "train.parquet", row_group_size=5)
    store.write_table(_series_frame(20), "val.parquet", row_group_size=5)
    index = MlDatasetIndex(
        pl.DataFrame(
            {
                BaseColumns.set_id: ["set-a", "set-b"],
                BaseColumns.cell_id: ["cell-a", "cell-b"],
                BaseColumns.cidx: [1, 2],
                BaseColumns.proto: [str(DatasetProtocolId.cycling)] * 2,
                BaseColumns.split: [BaseColumns.split.values.train, BaseColumns.split.values.val],
                BaseColumns.manifest: ["manifest.parquet", "manifest.parquet"],
                MANIFEST_ROW_ID_COLUMN: [0, 1],
                BaseColumns.row_n: [20, 20],
                BaseColumns.norm_segs: [
                    [_segment("train.parquet", 0, 20)],
                    [_segment("val.parquet", 0, 20)],
                ],
            }
        )
    )
    common = {
        "store": store,
        "index": index,
        "input_columns": (BaseColumns.time,),
        "target_columns": (BaseColumns.volt,),
        "scaling": (),
        "config": LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=3),
        ),
        "active_protocol": DatasetProtocolId.cycling,
    }

    train_loader = torch.utils.data.DataLoader(MlDataIterable(**common), batch_size=None)
    val_loader = dataloader_for_split(train_loader, BaseColumns.split.values.val)
    val_batch = next(iter(val_loader))

    assert val_batch.active.state.manifest_row_ids == (1,)


def test_loader_validates_only_active_split_protocol_columns(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(_series_frame(20), "cycling.parquet", row_group_size=5)
    store.write_table(
        pl.DataFrame({BaseColumns.freq: [1.0, 10.0], BaseColumns.z_real: [1.0, 2.0]}),
        "eis.parquet",
        row_group_size=5,
    )
    index = MlDatasetIndex(
        pl.DataFrame(
            {
                BaseColumns.set_id: ["set-a", "set-a"],
                BaseColumns.cell_id: ["cell-a", "cell-a"],
                BaseColumns.cidx: [1, 1],
                BaseColumns.proto: [str(DatasetProtocolId.cycling), str(DatasetProtocolId.eis)],
                BaseColumns.split: [BaseColumns.split.values.train] * 2,
                BaseColumns.manifest: ["manifest.parquet", "manifest.parquet"],
                MANIFEST_ROW_ID_COLUMN: [0, 1],
                BaseColumns.row_n: [20, 2],
                BaseColumns.norm_segs: [
                    [_segment("cycling.parquet", 0, 20)],
                    [_segment("eis.parquet", 0, 2)],
                ],
            }
        )
    )

    batch = next(
        iter(
            MlDataIterable(
                store=store,
                index=index,
                input_columns=(BaseColumns.time,),
                target_columns=(BaseColumns.volt,),
                scaling=(ScalingRule(BaseColumns.z_real, 0.0, 10.0),),
                config=LoaderConfig(
                    strategy="sequential",
                    default_window=WindowConfig(batch_size=1, seq_len=3),
                ),
                active_protocol=DatasetProtocolId.cycling,
            )
        )
    )

    assert tuple(batch.active.inputs.shape) == (1, 3, 1)


def test_create_dataloader_coerces_protocol_names(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    segment_path = "type=published/dataset=set-a/source=normalized/h.parquet"
    manifest_path = "type=published/dataset=set-a/source=normalized/manifest.parquet"
    store.write_table(_series_frame(20), segment_path, row_group_size=5)
    manifest = _index_for_store(segment_path, rows=20).frame.with_columns(
        pl.lit(str(DatasetProtocolId.hppc)).alias(BaseColumns.proto)
    )
    store.write_table(
        manifest.drop(MANIFEST_ROW_ID_COLUMN),
        manifest_path,
        metadata={
            str(BaseColumns.git_commit): "1234567890abcdef",
            str(BaseColumns.git_status): BaseColumns.git_status.values.clean,
        },
    )

    loader = create_dataloader(
        store,
        {manifest_path: "1234567"},
        input_columns=(BaseColumns.time,),
        target_columns=(BaseColumns.volt,),
        protocols=("hppc",),
        active_protocol="hppc",
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=3),
        ),
    )

    assert next(iter(loader)).active_protocol == DatasetProtocolId.hppc


def test_loader_rejects_missing_group_key_column(tmp_path: Path) -> None:
    index = _index_for_store("data.parquet", rows=20)
    index = MlDatasetIndex(index.frame.drop(BaseColumns.cell_id))

    with pytest.raises(ValueError, match="group/alignment columns"):
        MlDataIterable(
            store=RecordingStore(tmp_path / "store"),
            index=index,
            input_columns=(BaseColumns.time,),
            target_columns=(BaseColumns.volt,),
            scaling=(),
            config=LoaderConfig(),
            active_protocol=DatasetProtocolId.cycling,
        )


def test_loader_validates_scaling_from_manifest_stats(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(_series_frame(20), "data.parquet", row_group_size=5)
    index = _index_for_store("data.parquet", rows=20)
    index = MlDatasetIndex(
        index.frame.with_columns(
            pl.lit([{"column": BaseColumns.time, "min": 0.0, "max": 19.0}]).alias(
                BaseColumns.norm_stats
            )
        )
    )

    batch = next(
        iter(
            MlDataIterable(
                store=store,
                index=index,
                input_columns=(BaseColumns.time,),
                target_columns=(BaseColumns.volt,),
                scaling=(ScalingRule(BaseColumns.time, 0.0, 20.0),),
                config=LoaderConfig(
                    strategy="sequential",
                    default_window=WindowConfig(batch_size=1, seq_len=3),
                ),
                active_protocol=DatasetProtocolId.cycling,
            )
        )
    )

    assert tuple(batch.active.inputs.shape) == (1, 3, 1)
    assert store.slices[0][1] == ((0, 4),)


def test_loader_rejects_scaling_outside_manifest_stats(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(_series_frame(20), "data.parquet", row_group_size=5)
    index = _index_for_store("data.parquet", rows=20)
    index = MlDatasetIndex(
        index.frame.with_columns(
            pl.lit([{"column": BaseColumns.time, "min": 0.0, "max": 19.0}]).alias(
                BaseColumns.norm_stats
            )
        )
    )

    with pytest.raises(ValueError, match="Scaling bounds violated"):
        MlDataIterable(
            store=store,
            index=index,
            input_columns=(BaseColumns.time,),
            target_columns=(BaseColumns.volt,),
            scaling=(ScalingRule(BaseColumns.time, 0.0, 10.0),),
            config=LoaderConfig(),
            active_protocol=DatasetProtocolId.cycling,
        )


def test_loader_requires_manifest_stats_for_scaled_columns(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "store")
    store.write_table(_series_frame(20), "data.parquet", row_group_size=5)

    with pytest.raises(ValueError, match="normalized stats"):
        MlDataIterable(
            store=store,
            index=_index_for_store("data.parquet", rows=20),
            input_columns=(BaseColumns.time,),
            target_columns=(BaseColumns.volt,),
            scaling=(ScalingRule(BaseColumns.time, 0.0, 20.0),),
            config=LoaderConfig(),
            active_protocol=DatasetProtocolId.cycling,
        )


def test_minmax_scaling_inverse_tensor_uses_selected_columns() -> None:
    scaling = minmax_scaling({BaseColumns.crate: (-6.0, 6.0)})
    data = torch.tensor([[[10.0, -1.0], [20.0, 1.0]]])

    restored = inverse_scale_tensor(data, (BaseColumns.time, BaseColumns.crate), scaling)

    assert restored.tolist() == [[[10.0, -6.0], [20.0, 6.0]]]


def test_selected_preview_scaling_uses_explicit_notebook_rules() -> None:
    scaling = selected_preview_scaling((BaseColumns.crate, BaseColumns.volt, BaseColumns.dt))

    assert tuple(rule.name for rule in scaling) == (
        BaseColumns.crate,
        BaseColumns.volt,
        BaseColumns.dt,
    )
    assert scaling[0].input_min == -6.0
    assert scaling[0].input_max == 6.0
    assert scaling[2].transform == "log1p"


def test_selected_preview_scaling_requires_explicit_rules_for_all_columns() -> None:
    with pytest.raises(ValueError, match="Missing preview scaling rules"):
        selected_preview_scaling((BaseColumns.time, BaseColumns.crate))


def test_log1p_scaling_round_trips_tensor() -> None:
    scaling = (ScalingRule(BaseColumns.dt, 0.0, 10_000.0, transform="log1p"),)
    data = torch.tensor([[[0.0], [60.0], [10_000.0]]])

    scaled = scale_data(data, scaling)
    restored = inverse_scale_tensor(scaled, (BaseColumns.dt,), scaling)

    assert isinstance(scaled, torch.Tensor)
    assert torch.allclose(scaled[..., 0], torch.tensor([[-1.0, -0.1073, 1.0]]), atol=1e-4)
    assert torch.allclose(restored, data, atol=1e-3)


def test_batch_preview_submission_adds_scaling_when_enabled() -> None:
    selected_index = _index_for_store(
        "type=published/dataset=pozzato-2022/source=normalized/data.parquet", rows=20
    ).frame.with_columns(
        pl.lit("type=published/dataset=pozzato-2022/source=normalized/manifest.parquet").alias(
            BaseColumns.manifest
        )
    )

    submission = make_batch_preview_submission(
        submit_id=1,
        selected_index_frame=selected_index,
        batch_warning=None,
        input_columns=(BaseColumns.crate,),
        target_columns=(BaseColumns.volt,),
        batch_size=1,
        seq_len=3,
        batch_group_index=0,
        sample_index=0,
        consecutive_step=0,
        max_preview_group=0,
        max_sample_index=0,
        max_consecutive_index=0,
        strategy="sequential",
        stateful_n_windows=1,
        active_protocol=str(DatasetProtocolId.cycling),
        enable_scaling=True,
    )

    assert submission is not None
    assert {rule.name for rule in submission.scaling} == {BaseColumns.crate, BaseColumns.volt}


def _segment(path: str, row_start: int, row_count: int) -> dict[str, object]:
    return {
        str(BaseColumns.path): path,
        str(BaseColumns.row0): row_start,
        str(BaseColumns.row_n): row_count,
    }


def _series_frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            BaseColumns.time: list(range(rows)),
            BaseColumns.curr: [float(value) for value in range(rows)],
            BaseColumns.volt: [100.0 + value for value in range(rows)],
        }
    )


def _index_for_store(path: str, *, rows: int) -> MlDatasetIndex:
    return MlDatasetIndex(
        pl.DataFrame(
            {
                BaseColumns.set_id: ["set-a"],
                BaseColumns.cell_id: ["cell-a"],
                BaseColumns.cidx: [1],
                BaseColumns.proto: [str(DatasetProtocolId.cycling)],
                BaseColumns.split: [BaseColumns.split.values.train],
                BaseColumns.manifest: ["manifest.parquet"],
                MANIFEST_ROW_ID_COLUMN: [0],
                BaseColumns.row_n: [rows],
                BaseColumns.norm_segs: [[_segment(path, 0, rows)]],
            }
        )
    )
