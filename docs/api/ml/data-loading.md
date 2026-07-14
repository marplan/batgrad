# ML Data Loading

The loader consumes normalized manifests, preserves stream provenance, and
materializes one-row-ahead targets. Build an index separately when several
loaders or tools need to share the same validated split assignments.

Null input values are materialized as the `-2.0` sentinel. Null targets remain
`NaN` and are excluded by finite-target loss masking, while the batch mask records
source-row availability.

Sequential loading traverses requested protocols in order. Shuffled protocol
groups support cycling, HPPC, and RPT, but not EIS. Finite stateful groups discard
a final segment shorter than `stateful_n_windows`; whole-stream batches also
truncate longer lanes to the shortest lane.

::: batgrad.ml.data.index.available_manifest_paths
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.data.loader.create_dataloader
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.data.loader.create_index
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.data.loader.create_dataloader_from_index
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.data.index.MlDatasetIndex
    options:
      heading_level: 2
      members:
        - filter_split
      show_source: false

::: batgrad.ml.data.config.ValidationConfig
    options:
      heading_level: 2
      members:
        - sample
        - provide
        - merge
      show_source: false

::: batgrad.ml.data.config.WindowConfig
    options:
      heading_level: 2
      members:
        - step
        - window_rows
      show_source: false

::: batgrad.ml.data.config.LoaderConfig
    options:
      heading_level: 2
      members:
        - window_for
      show_source: false

::: batgrad.ml.data.config.ScalingRule
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.data.batch.Batch
    options:
      heading_level: 2
      members:
        - is_protocol
        - pin_memory
        - to
      show_source: false

::: batgrad.ml.data.batch.BatchState
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.data.batch.BatchSegmentRef
    options:
      heading_level: 2
      members: false
      show_source: false

## Scaling

::: batgrad.ml.data.scaling.scale_data
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.data.scaling.inverse_scale_data
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.data.scaling.inverse_scale_tensor
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.data.scaling.minmax_scaling
    options:
      heading_level: 3
      show_source: false
