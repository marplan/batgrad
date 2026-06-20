import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from batgrad.data.datasets.registry import get_dataset
    from batgrad.data.processing.normalize import NormalizeStageConfig
    from batgrad.contracts.mapping import BaseColumns, DatasetStageId
    from batgrad.viz.interactive import make_widgets
    from batgrad.logging import configure
    from batgrad.storage.local import LocalDataProcessingStore
    import marimo as mo
    import pyarrow.parquet as pq
    from batgrad.data.processing.normalize import (
        NormalizeProtocolSpec,
        NormalizeStageSpec,
    )
    from batgrad.contracts.protocols import BatteryProtocols, BatteryProtocolSpec

    from batgrad.data.datasets.pozzato_2022.config import (
        NORMALIZED_TIMESERIES,
        NORMALIZED_EIS,
        AMBIENT_TEMPERATURE_DEGC,
        CRateTransformSpec,
        NOMINAL_CAPACITY_AH,
    )


    from batgrad.data.transforms.resampling import (
        LinearResamplingSpec,
        MinMaxLTTBResamplingSpec,
        ResamplingSpec,
    )


    configure(level="INFO")
    return (
        AMBIENT_TEMPERATURE_DEGC,
        BaseColumns,
        BatteryProtocolSpec,
        BatteryProtocols,
        CRateTransformSpec,
        LinearResamplingSpec,
        LocalDataProcessingStore,
        MinMaxLTTBResamplingSpec,
        NOMINAL_CAPACITY_AH,
        NORMALIZED_EIS,
        NORMALIZED_TIMESERIES,
        NormalizeProtocolSpec,
        NormalizeStageConfig,
        NormalizeStageSpec,
        ResamplingSpec,
        get_dataset,
        make_widgets,
        mo,
        pl,
        pq,
    )


@app.cell
def _(LocalDataProcessingStore, get_dataset):
    dataset = get_dataset("pozzato-2022")
    input_store = LocalDataProcessingStore("/data/loc_datasets/")
    # manifest_path=input_store.resolve(dataset.spec.manifest(source=DatasetStageId.normalize))
    manifest_path = "/data/loc_datasets/type=published/dataset=pozzato-2022/source=ingested/manifest.parquet"
    shard_path = "/data/loc_datasets/type=published/dataset=pozzato-2022/source=ingested/cycling/cycling_part-000000.parquet"
    return dataset, input_store, manifest_path, shard_path


@app.cell
def _(mo, pq, shard_path):
    metadata = pq.read_metadata(shard_path)
    mo.ui.table(metadata.metadata)
    return


@app.cell
def _(manifest_path, mo, pl):
    mo.ui.table(
        pl.scan_parquet(manifest_path)
        .sort("cell id")
        .group_by(pl.col("cycle index"))
        .agg(pl.exclude("cycle index").explode())
        .sort("cycle index")
        .collect()
    )

    # _metadata = pq.read_metadata(manifest_path)
    # mo.ui.table(_metadata.metadata)
    # mo.ui.table(pl.scan_parquet(shard_path).head())
    return


@app.cell
def _(BaseColumns, NormalizeStageConfig, dataset, input_store):
    run = dataset.normalize_interactive(
        input_store,
        input_store,
        NormalizeStageConfig(
            n_jobs=-1,
            apply_resampling=False,
            max_batch_rows=500_000,
        ),
        protocols=["cycling", "EIS", "RPT", "HPPC"],
        group_values={BaseColumns.cidx: 1, BaseColumns.cell_id: "V4"},
        annotate=True,
    )
    return (run,)


@app.cell
def _(
    AMBIENT_TEMPERATURE_DEGC,
    BaseColumns,
    BatteryProtocolSpec,
    CRateTransformSpec,
    NOMINAL_CAPACITY_AH,
    NORMALIZED_TIMESERIES,
    NormalizeProtocolSpec,
    ResamplingSpec,
):
    from batgrad.data.datasets.pozzato_2022.config import TIME_NORMALIZE_CHECKS, ImpedanceComponentsCheckSpec, MissingCheckSpec, DomainAxisCheckSpec
    def time_normalize_protocol(
        protocol: BatteryProtocolSpec,
        *,
        resampling: ResamplingSpec | None = None,
    ) -> NormalizeProtocolSpec:
        return NormalizeProtocolSpec(
            protocol=protocol,
            columns=tuple(NORMALIZED_TIMESERIES),
            constant_columns={BaseColumns.amb_temp: AMBIENT_TEMPERATURE_DEGC},
            transforms=(
                CRateTransformSpec(
                    source_col=BaseColumns.curr,
                    target_col=BaseColumns.crate,
                    nominal_capacity_ah=NOMINAL_CAPACITY_AH,
                ),
            ),
            checks=TIME_NORMALIZE_CHECKS,
            resampling=resampling,
        )

    return (
        DomainAxisCheckSpec,
        ImpedanceComponentsCheckSpec,
        MissingCheckSpec,
        time_normalize_protocol,
    )


@app.cell
def _(
    AMBIENT_TEMPERATURE_DEGC,
    BaseColumns,
    BatteryProtocols,
    DomainAxisCheckSpec,
    ImpedanceComponentsCheckSpec,
    LinearResamplingSpec,
    MinMaxLTTBResamplingSpec,
    MissingCheckSpec,
    NORMALIZED_EIS,
    NormalizeProtocolSpec,
    NormalizeStageSpec,
    time_normalize_protocol,
):

    widget_normalize_spec = NormalizeStageSpec(
        protocol_specs=(
            time_normalize_protocol(
                protocol=BatteryProtocols.cyc,
                resampling=MinMaxLTTBResamplingSpec(
                    x_col=BaseColumns.time,
                    y_col=BaseColumns.volt,
                    # points_ratio=0.0001,
                    points=1000
                ),
            ),
            time_normalize_protocol(
                protocol=BatteryProtocols.hppc,
                resampling=MinMaxLTTBResamplingSpec(
                    x_col=BaseColumns.time,
                    y_col=BaseColumns.volt,
                    points=16_384,
                ),
            ),
            time_normalize_protocol(
                protocol=BatteryProtocols.rpt,
                resampling=MinMaxLTTBResamplingSpec(
                    x_col=BaseColumns.time,
                    y_col=BaseColumns.volt,
                    points=4096,
                ),
            ),
            NormalizeProtocolSpec(
                protocol=BatteryProtocols.eis,
                columns=tuple(NORMALIZED_EIS),
                constant_columns={BaseColumns.amb_temp: AMBIENT_TEMPERATURE_DEGC},
                checks=(
                    ImpedanceComponentsCheckSpec(),
                    MissingCheckSpec(),
                    DomainAxisCheckSpec(
                        axis_col=BaseColumns.freq,
                        zero_replacement=1e-7,
                        enforce_positive=True,
                    ),
                ),
                resampling=LinearResamplingSpec(
                    x_col=BaseColumns.freq,
                    points=48,
                    scale="log",
                ),
            ),
        ),
    )
    return (widget_normalize_spec,)


@app.cell
def _(
    BaseColumns,
    NormalizeStageConfig,
    dataset,
    input_store,
    run,
    widget_normalize_spec,
):
    widget_run = dataset.normalize_interactive(
        input_store,
        input_store,
        NormalizeStageConfig(
            n_jobs= -1,
            apply_resampling=False,
            max_batch_rows=500_000,
        ),
        protocols=["cycling","EIS", "RPT", "HPPC"],
        group_values={BaseColumns.cidx: 1, BaseColumns.cell_id: "V4"},
        annotate=True,
        source_run=run,
        normalize_spec=widget_normalize_spec
    )
    return (widget_run,)


@app.cell
def _(BaseColumns, make_widgets, mo, widget_run):
    widgets = make_widgets(
        widget_run,
        cols=(BaseColumns.volt, BaseColumns.crate, BaseColumns.temp),
        # overlay_sources=(DatasetStageId.ingested,),
        max_points_per_trace=1_000,
        max_points_per_figure=100_000,
        max_batch_rows=500_000,
    )
    mo.vstack([mo.ui.anywidget(widget.show()) for widget in widgets])
    return


@app.cell
def _(run):
    run.clean()
    return


if __name__ == "__main__":
    app.run()
