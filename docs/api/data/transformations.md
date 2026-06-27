# Transformations

## Transforms

::: batgrad.data.transforms.transforms.CRateTransformSpec
    options:
      heading_level: 3
      members:
        - input_columns
        - produced_columns
        - apply
      show_source: false

## Checks

::: batgrad.data.transforms.checks.MissingCheckSpec
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.data.transforms.checks.TimeCheckSpec
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.data.transforms.checks.ColumnBoundsCheckSpec
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.data.transforms.checks.ImpedanceComponentsCheckSpec
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.data.transforms.checks.DomainAxisCheckSpec
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.data.transforms.checks.apply_checks_full_task
    options:
      heading_level: 3
      show_source: false

::: batgrad.data.transforms.checks.apply_checks_bounded_chunk
    options:
      heading_level: 3
      show_source: false

## Resampling

::: batgrad.data.transforms.resampling.MinMaxLTTBResamplingSpec
    options:
      heading_level: 3
      members:
        - input_columns
        - metadata_values
      show_source: false

::: batgrad.data.transforms.resampling.LinearResamplingSpec
    options:
      heading_level: 3
      members:
        - input_columns
        - metadata_values
      show_source: false

::: batgrad.data.transforms.resampling.resolve_min_max_lttb_budget
    options:
      heading_level: 3
      show_source: false
