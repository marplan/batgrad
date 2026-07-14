from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.config import (
    ScalingRuleConfig,
    ValidationGroupConfig,
    ValidationSplitConfig,
)
from batgrad.ml.data.config import LoaderConfig, ScalingRule, ValidationConfig, WindowConfig
from batgrad.ml.data.loader import create_dataloader, create_index
from batgrad.ml.experiment import data_validation_config, scaling_rules, val_loader_config
from batgrad.ml.validation import run_rollouts
from tests.ml.conftest import (
    INPUT_COLUMNS,
    TARGET_COLUMNS,
    TINY_GIT_COMMIT,
    TINY_MANIFEST_PATH,
    InMemoryMlStore,
    RecordingModel,
    make_config,
    make_memory_manifest_store,
    manifest_footer_bytes,
    series_frame,
    shard_path,
)

MANIFEST_PATH = "type=synthetic/dataset=tiny-ml/source=normalized/manifest.parquet"
GIT_COMMIT = "abcdef0"


@pytest.fixture
def tiny_manifest_store() -> InMemoryMlStore:
    return make_memory_manifest_store()


def test_create_dataloader_from_synthetic_manifest_materializes_batches(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=3, seq_len=10),
        ),
    )

    batch = next(iter(loader))

    assert tuple(batch.inputs.shape) == (3, 10, len(INPUT_COLUMNS))
    assert tuple(batch.targets.shape) == (3, 10, len(TARGET_COLUMNS))
    assert tuple(batch.mask.shape) == (3, 10)
    assert batch.all_valid is True
    assert batch.state.manifest_paths == (MANIFEST_PATH,)
    assert batch.state.group_keys[0] == ("tiny-ml", "cell-a", 1, "cycling")
    assert batch.state.window_offsets == (0,)
    flat_inputs = batch.inputs.reshape(-1, len(INPUT_COLUMNS))
    flat_targets = batch.targets.reshape(-1, len(TARGET_COLUMNS))
    assert torch.equal(flat_targets[:-1, 0], flat_inputs[1:, 2])


@pytest.mark.parametrize("data_access", ["windowed", "full_in_mem"])
def test_loader_preserves_null_targets_as_non_finite(
    tiny_manifest_store: InMemoryMlStore,
    data_access: str,
) -> None:
    shard = shard_path("cell-a", 1, DatasetProtocolId.cycling, dataset="tiny-ml")
    tiny_manifest_store.tables[shard] = tiny_manifest_store.tables[shard].with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(None)
        .otherwise(pl.col("voltage"))
        .alias("voltage")
    )
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=10),
            data_access=data_access,
        ),
    )

    batch = next(iter(loader))

    assert batch.inputs[0, 1, 2].item() == -2.0
    assert torch.isnan(batch.targets[0, 0, 0])
    assert batch.mask[0, 0].item() is True


def test_manifest_loader_shuffled_protocol_groups_preserves_stateful_sequence(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="shuffled_protocol_groups",
            protocol_order=(DatasetProtocolId.cycling,),
            default_window=WindowConfig(batch_size=3, seq_len=10),
            stateful_n_windows=2,
        ),
    )

    first, second = [batch for _, batch in zip(range(2), iter(loader), strict=False)]

    assert first.state.stateful_group_idx == second.state.stateful_group_idx
    assert first.state.stateful_step_idx == 0
    assert second.state.stateful_step_idx == 1
    assert first.state.stateful_steps == second.state.stateful_steps == 2
    assert first.state.alignment_keys == second.state.alignment_keys
    assert tuple(
        second_offset - first_offset
        for first_offset, second_offset in zip(
            first.state.window_offsets, second.state.window_offsets, strict=True
        )
    ) == (10, 10, 10)


def test_manifest_loader_cross_protocol_chain_keeps_same_alignment_key(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling, DatasetProtocolId.hppc),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="shuffled_protocol_groups",
            protocol_order=(DatasetProtocolId.cycling, DatasetProtocolId.hppc),
            default_window=WindowConfig(batch_size=1, seq_len=10),
            stateful_n_windows=-1,
            cross_protocol_state_carry="chain",
        ),
    )

    batches = [batch for _, batch in zip(range(8), iter(loader), strict=False)]

    assert len({batch.state.stateful_group_idx for batch in batches}) == 1
    assert [batch.state.stateful_step_idx for batch in batches] == list(range(8))
    assert [batch.state.protocols[0] for batch in batches] == [
        DatasetProtocolId.cycling,
        DatasetProtocolId.cycling,
        DatasetProtocolId.cycling,
        DatasetProtocolId.cycling,
        DatasetProtocolId.hppc,
        DatasetProtocolId.hppc,
        DatasetProtocolId.hppc,
        DatasetProtocolId.hppc,
    ]
    assert len({batch.state.alignment_keys[0] for batch in batches}) == 1


def test_manifest_loader_reads_window_across_multiple_segments() -> None:
    first_shard = "type=synthetic/dataset=tiny-ml/source=normalized/cell=cell-a/part=0.parquet"
    second_shard = "type=synthetic/dataset=tiny-ml/source=normalized/cell=cell-a/part=1.parquet"
    store = InMemoryMlStore(
        {
            first_shard: series_frame(20, offset=0),
            second_shard: series_frame(24, offset=20),
            MANIFEST_PATH: pl.DataFrame(
                [
                    {
                        BaseColumns.set_id: "tiny-ml",
                        BaseColumns.cell_id: "cell-a",
                        BaseColumns.cidx: 1,
                        BaseColumns.proto: str(DatasetProtocolId.cycling),
                        BaseColumns.row_n: 44,
                        BaseColumns.norm_stats: [
                            {"column": column, "min": 0.0, "max": 2000.0}
                            for column in INPUT_COLUMNS
                        ],
                        BaseColumns.norm_segs: [
                            {
                                str(BaseColumns.path): first_shard,
                                str(BaseColumns.row0): 0,
                                str(BaseColumns.row_n): 20,
                            },
                            {
                                str(BaseColumns.path): second_shard,
                                str(BaseColumns.row0): 0,
                                str(BaseColumns.row_n): 24,
                            },
                        ],
                    }
                ]
            ),
        },
        {
            MANIFEST_PATH: manifest_footer_bytes(
                {
                    str(BaseColumns.git_commit): GIT_COMMIT,
                    str(BaseColumns.git_status): str(BaseColumns.git_status.values.clean),
                }
            )
        },
    )
    loader = create_dataloader(
        store=store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=10, step_rows=15),
        ),
    )

    _first_batch, crossing_batch = [batch for _, batch in zip(range(2), iter(loader), strict=False)]

    assert crossing_batch.state.window_offsets == (15,)
    assert crossing_batch.all_valid is True
    assert crossing_batch.inputs[0, :, 0].tolist() == [float(idx) for idx in range(15, 25)]
    assert crossing_batch.targets[0, :, 0].tolist() == [
        pytest.approx(3.0 + (idx % 10) / 100.0) for idx in range(16, 26)
    ]
    assert [segment.path for segment in crossing_batch.state.segments] == [
        first_shard,
        second_shard,
    ]
    assert [segment.window_row_count for segment in crossing_batch.state.segments] == [5, 6]

    full_in_mem_loader = create_dataloader(
        store=store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=10, step_rows=15),
            data_access="full_in_mem",
        ),
    )
    _first_cached, crossing_cached = [
        batch for _, batch in zip(range(2), iter(full_in_mem_loader), strict=False)
    ]

    assert torch.equal(crossing_cached.inputs, crossing_batch.inputs)
    assert torch.equal(crossing_cached.targets, crossing_batch.targets)
    assert torch.equal(crossing_cached.mask, crossing_batch.mask)


def test_manifest_validation_split_provide_selects_cell_cycle_protocol(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    validation = ValidationConfig.provide(
        (
            {
                BaseColumns.set_id: "tiny-ml",
                BaseColumns.cell_id: "cell-b",
                BaseColumns.cidx: 2,
                BaseColumns.proto: str(DatasetProtocolId.cycling),
            },
        ),
        group_by=(BaseColumns.set_id, BaseColumns.cell_id, BaseColumns.cidx, BaseColumns.proto),
    )
    common = {
        "store": tiny_manifest_store,
        "manifest_paths": {MANIFEST_PATH: GIT_COMMIT},
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "protocols": (DatasetProtocolId.cycling,),
        "validation": validation,
    }

    train_batch = next(
        iter(
            create_dataloader(
                config=LoaderConfig(
                    split=BaseColumns.split.values.train,
                    strategy="sequential",
                    default_window=WindowConfig(batch_size=1, seq_len=10),
                ),
                **common,
            )
        )
    )
    val_batch = next(
        iter(
            create_dataloader(
                config=LoaderConfig(
                    split=BaseColumns.split.values.val,
                    strategy="sequential",
                    default_window=WindowConfig(batch_size=1, seq_len=10),
                ),
                **common,
            )
        )
    )

    assert train_batch.state.group_keys[0][:3] != ("tiny-ml", "cell-b", 2)
    assert val_batch.state.group_keys[0] == ("tiny-ml", "cell-b", 2, "cycling")


def test_manifest_validation_split_merge_keeps_provided_group_and_samples_more(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    validation = ValidationConfig.merge(
        (
            {
                BaseColumns.set_id: "tiny-ml",
                BaseColumns.cell_id: "cell-b",
                BaseColumns.cidx: 2,
                BaseColumns.proto: str(DatasetProtocolId.cycling),
            },
        ),
        fraction=0.5,
        group_by=(BaseColumns.set_id, BaseColumns.cell_id, BaseColumns.cidx, BaseColumns.proto),
    )

    index = create_index(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        protocols=(DatasetProtocolId.cycling, DatasetProtocolId.hppc),
        validation=validation,
    )

    val_rows = index.frame.filter(pl.col(BaseColumns.split) == BaseColumns.split.values.val)
    assert val_rows.height == 4
    assert (
        val_rows.filter(
            (pl.col(BaseColumns.cell_id) == "cell-b")
            & (pl.col(BaseColumns.cidx) == 2)
            & (pl.col(BaseColumns.proto) == str(DatasetProtocolId.cycling))
        ).height
        == 1
    )


def test_manifest_split_resolves_rollout_selector_and_anchor(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    config = make_config(batch_size=1, seq_len=10, suffix_steps=3)
    selector = {
        BaseColumns.set_id: "tiny-ml",
        BaseColumns.cell_id: "cell-b",
        BaseColumns.cidx: 2,
        BaseColumns.proto: str(DatasetProtocolId.cycling),
    }
    config = replace(
        config,
        data=replace(
            config.data,
            manifest_paths={TINY_MANIFEST_PATH: TINY_GIT_COMMIT},
            scaling=tuple(
                ScalingRuleConfig(column=column, input_min=0.0, input_max=2000.0)
                for column in INPUT_COLUMNS
            ),
        ),
        validation=replace(
            config.validation,
            split=ValidationSplitConfig(
                strategy="provide",
                group_by=(
                    BaseColumns.set_id,
                    BaseColumns.cell_id,
                    BaseColumns.cidx,
                    BaseColumns.proto,
                ),
                groups=(ValidationGroupConfig(match=selector, rollout_start_offsets=(12,)),),
            ),
            max_tf_batches=0,
            rollout_steps=3,
        ),
    )
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths=config.data.manifest_paths,
        input_columns=config.data.input_columns,
        target_columns=config.data.target_columns,
        protocols=config.data.protocols,
        validation=data_validation_config(config),
        scaling=scaling_rules(config),
        config=val_loader_config(config),
    )
    dataset = loader.dataset

    result = run_rollouts(
        config,
        RecordingModel(),
        dataset.full_index,
        tiny_manifest_store,
        torch.device("cpu"),
    )

    assert result.rollout_metrics is not None
    assert tiny_manifest_store.slices[-1][1][0][0] == 3


def test_full_in_mem_loader_matches_windowed_loader_from_manifest(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    common = {
        "store": tiny_manifest_store,
        "manifest_paths": {MANIFEST_PATH: GIT_COMMIT},
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "protocols": (DatasetProtocolId.cycling,),
        "validation": ValidationConfig.sample(fraction=0.0),
    }
    window = WindowConfig(batch_size=3, seq_len=10)

    windowed = next(
        iter(
            create_dataloader(
                config=LoaderConfig(strategy="sequential", default_window=window),
                **common,
            )
        )
    )
    full_in_mem = next(
        iter(
            create_dataloader(
                config=LoaderConfig(
                    strategy="sequential",
                    default_window=window,
                    data_access="full_in_mem",
                ),
                **common,
            )
        )
    )

    assert torch.equal(full_in_mem.inputs, windowed.inputs)
    assert torch.equal(full_in_mem.targets, windowed.targets)
    assert torch.equal(full_in_mem.mask, windowed.mask)


def test_full_in_mem_matches_windowed_for_shuffled_stateful_protocol_chain(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    common = {
        "store": tiny_manifest_store,
        "manifest_paths": {MANIFEST_PATH: GIT_COMMIT},
        "input_columns": INPUT_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "protocols": (DatasetProtocolId.cycling, DatasetProtocolId.hppc),
        "validation": ValidationConfig.sample(fraction=0.0),
        "scaling": (ScalingRule(column="time", input_min=0.0, input_max=2000.0),),
    }
    config = LoaderConfig(
        strategy="shuffled_protocol_groups",
        protocol_order=(DatasetProtocolId.cycling, DatasetProtocolId.hppc),
        default_window=WindowConfig(batch_size=1, seq_len=10),
        stateful_n_windows=-1,
        cross_protocol_state_carry="chain",
    )
    windowed = list(iter(create_dataloader(config=config, **common)))
    full_in_mem = list(
        iter(create_dataloader(config=replace(config, data_access="full_in_mem"), **common))
    )

    assert len(full_in_mem) == len(windowed)
    for cached, streamed in zip(full_in_mem, windowed, strict=True):
        assert torch.equal(cached.inputs, streamed.inputs)
        assert torch.equal(cached.targets, streamed.targets)
        assert torch.equal(cached.mask, streamed.mask)
        assert cached.state == streamed.state


def test_manifest_loader_drop_incomplete_false_materializes_padded_tail(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=3, seq_len=10, drop_incomplete=False),
        ),
    )

    _first, tail = [batch for _, batch in zip(range(2), iter(loader), strict=False)]

    assert tail.state.window_offsets == (30,)
    assert tail.all_valid is False
    assert tail.mask.reshape(-1).tolist()[:13] == [True] * 13
    assert tail.mask.reshape(-1).tolist()[13:] == [False] * 17


def test_protocol_mode_available_keeps_existing_protocols(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling, DatasetProtocolId.rpt),
        protocol_mode="available",
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=10),
        ),
    )

    assert next(iter(loader)).is_protocol(DatasetProtocolId.cycling)

    with pytest.raises(ValueError, match="Requested protocols not found"):
        create_dataloader(
            store=tiny_manifest_store,
            manifest_paths={MANIFEST_PATH: GIT_COMMIT},
            input_columns=INPUT_COLUMNS,
            target_columns=TARGET_COLUMNS,
            protocols=(DatasetProtocolId.cycling, DatasetProtocolId.rpt),
            protocol_mode="strict",
            validation=ValidationConfig.sample(fraction=0.0),
            config=LoaderConfig(
                strategy="sequential",
                default_window=WindowConfig(batch_size=1, seq_len=10),
            ),
        )


def test_manifest_loader_applies_scaling_rules(
    tiny_manifest_store: InMemoryMlStore,
) -> None:
    loader = create_dataloader(
        store=tiny_manifest_store,
        manifest_paths={MANIFEST_PATH: GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        scaling=(ScalingRule(column="time", input_min=0.0, input_max=2000.0),),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=10),
        ),
    )

    batch = next(iter(loader))

    assert batch.inputs[0, 0, 0].item() == pytest.approx(-0.9)
