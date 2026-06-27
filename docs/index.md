# batgrad

`batgrad` provides data processing tools for battery degradation datasets.

```mermaid
flowchart TB
  dataset["mapping.py<br/>config.py<br/>raw.py"]

  raw_adapter{{"RawDatasetAdapter<br/>plan_raw_tasks<br/>load_raw_task"}}
  ingest_method{{"Dataset.ingest"}}
  normalize{{"transforms<br/>checks<br/>resampling"}}
  normalize_method{{"Dataset.normalize"}}
  interactive_method{{"Dataset.normalize_interactive"}}

  subgraph raw_stage["stage: raw"]
    raw_files[("Raw source files")]
  end

  subgraph ingested_stage["stage: ingested"]
    ingested[("Ingested parquet<br/>shards + manifest")]
  end

  subgraph normalized_stage["stage: normalized"]
    normalized[("Normalized parquet<br/>shards + manifest")]
    scratch[("Scratch parquet<br/>shards + manifest")]
  end

  dataset --> raw_adapter
  dataset --> normalize

  raw_files --> raw_adapter
  raw_adapter --> ingest_method --> ingested

  ingested --> normalize
  normalize --> normalize_method --> normalized
  normalize --> interactive_method --> scratch

  click raw_adapter "api/data/ingested/#batgrad.data.processing.raw.RawDatasetAdapter" "RawDatasetAdapter API"
  click ingest_method "api/data/configuration/#batgrad.data.datasets.config.Dataset.ingest" "Dataset.ingest API"
  click normalize "api/data/transformations/" "Transformations API"
  click normalize_method "api/data/configuration/#batgrad.data.datasets.config.Dataset.normalize" "Dataset.normalize API"
  click interactive_method "api/data/configuration/#batgrad.data.datasets.config.Dataset.normalize_interactive" "Dataset.normalize_interactive API"
```

Start with [Quick Start](quick-start.md). Use
[Environment Setup](environment-setup.md) for container details and the API
Reference for implementation details.
