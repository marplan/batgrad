from __future__ import annotations

from batgrad.data.datasets.pozzato_2022.mapping import Pozzato2022Columns
from batgrad.data.datasets.specs import (
    DatasetInfo,
    DatasetSpec,
    NormalizeSpec,
    RawIngestSpec,
    TokenNormalizeSpec,
)
from batgrad.data.locations import DatasetLocation

cols = Pozzato2022Columns

RAW_INGEST_SPEC = RawIngestSpec(
    file_suffixes=(".xlsx",),
    row_group_size=256_000,
    max_shard_size_bytes=500 * 1024 * 1024,
    shard_size_tolerance_ratio=0.03,
)


NORMALIZE_SPEC = NormalizeSpec(
    spec_id="pozzato-2022-normalized-v1",
    token_specs={
        "Cycling": TokenNormalizeSpec(
            columns=(
                cols.dt,
                cols.c_rate,
                cols.voltage,
                cols.temperature,
            ),
            checks=("missing", "time", "battery_signal_corr"),
            resampling_profile_id="cycling_default_v1",
            scaling_profile_id="battery_timeseries_v1",
        ),
        "HPPC": TokenNormalizeSpec(
            columns=(
                cols.dt,
                cols.c_rate,
                cols.voltage,
                cols.temperature,
            ),
            checks=("missing", "time", "battery_signal_corr"),
            resampling_profile_id="hppc_default_v1",
            scaling_profile_id="battery_timeseries_v1",
        ),
    },
)


DATASET_SPEC = DatasetSpec(
    dataset_id="pozzato-2022",
    location=DatasetLocation(
        dataset_type="published",
        dataset_id="pozzato-2022",
    ),
    info=DatasetInfo(
        name="Pozzato 2022",
        year=2022,
        author="Pozzato",
        misc={"Chemistry": "NMC/graphite"},
    ),
    raw=RAW_INGEST_SPEC,
    normalize=NORMALIZE_SPEC,
    cols=Pozzato2022Columns,
)
