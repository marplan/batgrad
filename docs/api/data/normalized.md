# Normalized

::: batgrad.data.processing.normalize.run_normalize
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.processing.normalize.run_normalize_interactive
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.processing.normalize.plan_normalize_tasks
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.processing.normalize.normalize_spec_with_resampling
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.processing.normalize.NormalizeStageConfig
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.data.processing.normalize.NormalizeStageSpec
    options:
      heading_level: 2
      members:
        - protocol_spec
        - output_columns
        - required_input_columns
        - task_metadata
        - output_spec
      show_source: false

::: batgrad.data.processing.normalize.NormalizeProtocolSpec
    options:
      heading_level: 2
      members:
        - protocol_id
        - group_by
        - output_columns
        - required_input_columns
      show_source: false

::: batgrad.data.processing.normalize.NormalizeTask
    options:
      heading_level: 2
      members: false
      show_source: false
