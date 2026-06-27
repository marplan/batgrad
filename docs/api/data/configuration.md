# Configuration

::: batgrad.data.datasets.registry.get_dataset
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.datasets.config.DatasetInfo
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.data.datasets.config.DatasetSpec
    options:
      heading_level: 2
      members:
        - root
        - source_root
        - source_file
        - manifest
      show_source: false

::: batgrad.data.datasets.config.Dataset
    options:
      heading_level: 2
      members:
        - ingest
        - normalize
        - normalize_interactive
        - load_interactive
        - load_interactive_manifest
      show_source: false

::: batgrad.data.processing.interactive.InteractiveStageRun
    options:
      heading_level: 2
      members:
        - manifest
        - protocol_spec
        - scan
        - clean
        - iter_sources
      show_source: false
