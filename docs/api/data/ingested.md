# Ingested

::: batgrad.data.processing.raw.run_ingest
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.processing.raw.IngestStageConfig
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.data.processing.raw.IngestStageSpec
    options:
      heading_level: 2
      members:
        - protocol_spec
        - output_columns
        - required_metadata
        - manifest_columns
        - is_included_file
        - output_spec
      show_source: false

::: batgrad.data.processing.raw.IngestProtocolSpec
    options:
      heading_level: 2
      members:
        - protocol_id
        - protocol_metadata
        - output_columns
        - manifest_columns
        - required_metadata
      show_source: false

::: batgrad.data.processing.raw.RawDatasetAdapter
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.processing.raw.IngestTask
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.data.processing.raw.IngestBatch
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.data.processing.raw.prepare_raw_batch
    options:
      heading_level: 2
      show_source: false

::: batgrad.data.processing.raw.align_to_protocol_spec
    options:
      heading_level: 2
      show_source: false
