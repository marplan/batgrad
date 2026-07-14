from __future__ import annotations

from dataclasses import replace

import polars as pl

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.contracts.row_ids import MANIFEST_ROW_ID_COLUMN
from batgrad.ml.data.config import LoaderConfig, WindowConfig
from batgrad.ml.data.index import MlDatasetIndex
from batgrad.ml.data.loader import MlDataIterable
from batgrad.ml.data.planning import (
    build_batch_plans,
    count_batch_plans,
)
from tests.ml.conftest import INPUT_COLUMNS, TARGET_COLUMNS, make_index, make_store


def test_shuffled_protocol_epoch_phase_changes_offsets() -> None:
    config = LoaderConfig(
        strategy="shuffled_protocol_groups",
        default_window=WindowConfig(batch_size=1, seq_len=10),
        protocol_order=(DatasetProtocolId.cycling,),
        stateful_n_windows=1,
    )
    dataset = MlDataIterable(
        store=make_store(rows=80),
        index=make_index(rows=80, split_cell_b=False),
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        scaling=(),
        config=config,
    )

    epoch0_offsets = {batch.state.window_offsets for batch in dataset}
    dataset.set_epoch(1)
    epoch1_offsets = {batch.state.window_offsets for batch in dataset}

    assert epoch0_offsets != epoch1_offsets


def test_shuffled_protocol_schedule_is_reproducible_for_seed_and_epoch() -> None:
    index = make_index(rows=80, split_cell_b=False)

    def schedule(seed: int) -> tuple[tuple[tuple[object, ...], int], ...]:
        config = LoaderConfig(
            strategy="shuffled_protocol_groups",
            default_window=WindowConfig(batch_size=3, seq_len=10),
            protocol_order=(DatasetProtocolId.cycling,),
            stateful_n_windows=1,
            seed=seed,
        )
        return tuple(
            (ref.stream_identity, ref.offset)
            for plan in build_batch_plans(index, config, epoch_idx=2)
            for ref in plan.refs
        )

    assert schedule(69) == schedule(69)
    assert schedule(69) != schedule(70)


def test_sequential_schedule_traverses_every_requested_protocol() -> None:
    index = make_index(rows=35, split_cell_b=False)
    config = LoaderConfig(
        strategy="sequential",
        default_window=WindowConfig(batch_size=1, seq_len=10),
        protocol_order=(DatasetProtocolId.cycling, DatasetProtocolId.hppc),
    )

    plans = build_batch_plans(index, config)

    assert len(plans) == count_batch_plans(index, config)
    assert {plan.refs[0].protocol for plan in plans} == {
        DatasetProtocolId.cycling,
        DatasetProtocolId.hppc,
    }
    protocol_sequence = tuple(plan.refs[0].protocol for plan in plans)
    first_hppc = protocol_sequence.index(DatasetProtocolId.hppc)
    assert all(protocol == DatasetProtocolId.cycling for protocol in protocol_sequence[:first_hppc])


def test_cross_protocol_state_carry_chains_by_alignment_key() -> None:
    index = make_index(rows=35, split_cell_b=False)
    config = LoaderConfig(
        strategy="shuffled_protocol_groups",
        default_window=WindowConfig(batch_size=1, seq_len=10),
        protocol_order=(DatasetProtocolId.cycling, DatasetProtocolId.hppc),
        stateful_n_windows=-1,
        cross_protocol_state_carry="chain",
    )

    plans = build_batch_plans(index, config, epoch_idx=0)
    by_group: dict[int, list[object]] = {}
    for plan in plans:
        assert plan.stateful_group_idx is not None
        by_group.setdefault(plan.stateful_group_idx, []).extend(plan.refs)
    chained = next(refs for refs in by_group.values() if len({ref.protocol for ref in refs}) > 1)

    assert [ref.protocol for ref in chained] == [
        DatasetProtocolId.cycling,
        DatasetProtocolId.cycling,
        DatasetProtocolId.cycling,
        DatasetProtocolId.hppc,
        DatasetProtocolId.hppc,
        DatasetProtocolId.hppc,
    ]
    assert len({ref.alignment_key for ref in chained}) == 1


def test_whole_stream_stateful_batches_warn_and_count_truncated_lanes(caplog) -> None:
    records = []
    for row_id, (cell, rows) in enumerate((("cell-a", 35), ("cell-b", 55))):
        records.append(
            {
                MANIFEST_ROW_ID_COLUMN: row_id,
                BaseColumns.set_id: "synthetic-ml",
                BaseColumns.cell_id: cell,
                BaseColumns.cidx: 1,
                BaseColumns.proto: str(DatasetProtocolId.cycling),
                BaseColumns.split: BaseColumns.split.values.train,
                BaseColumns.manifest: "memory-manifest.parquet",
                BaseColumns.row_n: rows,
                BaseColumns.norm_segs: [
                    {
                        str(BaseColumns.path): f"memory/{cell}.parquet",
                        str(BaseColumns.row0): 0,
                        str(BaseColumns.row_n): rows,
                    }
                ],
            }
        )
    index = MlDatasetIndex(pl.DataFrame(records))
    config = LoaderConfig(
        strategy="shuffled_protocol_groups",
        default_window=WindowConfig(batch_size=2, seq_len=10),
        protocol_order=(DatasetProtocolId.cycling,),
        stateful_n_windows=-1,
    )

    with caplog.at_level("WARNING", logger="batgrad.ml.data.planning"):
        plans = build_batch_plans(index, config, epoch_idx=0)

    assert len(plans) == 3
    assert [plan.stateful_step_idx for plan in plans] == [0, 1, 2]
    assert all(len(plan.refs) == 2 for plan in plans)
    assert "dropped_windows=2" in caplog.text
    assert count_batch_plans(index, config, epoch_idx=0) == len(plans)


def test_steps_per_epoch_matches_planned_batches() -> None:
    config = LoaderConfig(
        strategy="shuffled_protocol_groups",
        default_window=WindowConfig(batch_size=3, seq_len=10),
        protocol_order=(DatasetProtocolId.cycling,),
        stateful_n_windows=2,
    )
    dataset = MlDataIterable(
        store=make_store(rows=80),
        index=make_index(rows=80, split_cell_b=False),
        input_columns=INPUT_COLUMNS,
        target_columns=TARGET_COLUMNS,
        scaling=(),
        config=config,
    )

    assert dataset.steps_per_epoch(0) == len(
        build_batch_plans(
            dataset.index,
            replace(config, protocol_order=dataset.protocol_order),
        )
    )
