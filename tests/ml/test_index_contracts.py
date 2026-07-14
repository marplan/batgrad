from __future__ import annotations

import polars as pl
import pytest
import torch

from batgrad.contracts.mapping import BaseColumns, DatasetProtocolId
from batgrad.ml.data.config import LoaderConfig, ValidationConfig, WindowConfig
from batgrad.ml.data.loader import create_dataloader, create_index
from tests.ml.conftest import (
    INPUT_COLUMNS,
    TINY_GIT_COMMIT,
    TINY_MANIFEST_PATH,
    InMemoryMlStore,
    make_memory_manifest_store,
    manifest_footer_bytes,
)


def test_manifest_index_infers_dataset_id_when_manifest_column_is_absent() -> None:
    store = make_memory_manifest_store()
    store.tables[TINY_MANIFEST_PATH] = store.tables[TINY_MANIFEST_PATH].drop(BaseColumns.set_id)

    index = create_index(
        store,
        {TINY_MANIFEST_PATH: TINY_GIT_COMMIT},
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
    )

    assert index.frame[BaseColumns.set_id].unique().to_list() == ["tiny-ml"]


def test_manifest_index_rejects_conflicting_dataset_ids() -> None:
    store = make_memory_manifest_store()
    store.tables[TINY_MANIFEST_PATH] = store.tables[TINY_MANIFEST_PATH].with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("other-dataset"))
        .otherwise(pl.col(BaseColumns.set_id))
        .alias(BaseColumns.set_id)
    )

    with pytest.raises(ValueError, match="multiple dataset ids"):
        create_index(store, {TINY_MANIFEST_PATH: TINY_GIT_COMMIT})


def test_manifest_index_rejects_unmatched_exact_cell_selector() -> None:
    store = make_memory_manifest_store()
    validation = ValidationConfig.provide(
        ({BaseColumns.cell_id: "unknown-cell"},),
        group_by=(BaseColumns.set_id, BaseColumns.cell_id),
    )

    with pytest.raises(ValueError, match="did not match any group"):
        create_index(store, {TINY_MANIFEST_PATH: TINY_GIT_COMMIT}, validation=validation)


def test_manifest_index_warns_for_dirty_footer_but_loads(caplog) -> None:
    store = make_memory_manifest_store()
    dirty_store = InMemoryMlStore(
        store.tables,
        {
            TINY_MANIFEST_PATH: manifest_footer_bytes(
                {
                    str(BaseColumns.git_commit): TINY_GIT_COMMIT,
                    str(BaseColumns.git_status): "dirty",
                }
            ),
        },
    )

    index = create_index(dirty_store, {TINY_MANIFEST_PATH: TINY_GIT_COMMIT})

    assert index.frame.height > 0
    assert "dirty git worktree" in caplog.text


def test_manifest_index_rejects_noncanonical_paths_before_store_access() -> None:
    with pytest.raises(ValueError, match="canonical"):
        create_index(InMemoryMlStore(), {"manifest.parquet": TINY_GIT_COMMIT})


def test_loader_uses_input_padding_and_preserves_null_targets() -> None:
    store = make_memory_manifest_store(rows=12)
    shard = next(path for path in store.tables if path.endswith("protocol=cycling.parquet"))
    store.tables[shard] = store.tables[shard].with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("voltage"))
        .alias("voltage")
    )

    loader = create_dataloader(
        store=store,
        manifest_paths={TINY_MANIFEST_PATH: TINY_GIT_COMMIT},
        input_columns=INPUT_COLUMNS,
        target_columns=("voltage", "temperature"),
        protocols=(DatasetProtocolId.cycling,),
        validation=ValidationConfig.sample(fraction=0.0),
        config=LoaderConfig(
            strategy="sequential",
            default_window=WindowConfig(batch_size=1, seq_len=10),
        ),
    )

    batch = next(iter(loader))

    assert batch.mask[0].all()
    assert batch.inputs[0, 2, 2].item() == -2.0
    assert torch.isnan(batch.targets[0, 1, 0])
