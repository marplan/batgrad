# Quick Start

## Environment

Start the local dev container and run the project setup script:

```sh
cp .env.example .env
docker compose up -d --build dev
docker compose exec -it dev zsh
./scripts/setup_project.sh
```

For remote containers, provider images, data mounts, and environment variables,
see [Environment Setup](environment-setup.md).

## Data Processing

The data flow is `raw files -> ingested parquet -> normalized parquet`. Interactive
runs write scratch outputs for inspection without replacing persisted stage data.

### Add a Dataset

Follow the current `pozzato_2022` dataset layout:

```text
batgrad/data/datasets/<dataset_id>/
  __init__.py
  mapping.py
  raw.py
  config.py
```

- `mapping.py`: define dataset-specific raw aliases and canonical columns with `MappingSpec`.
- `raw.py`: implement a raw adapter with `plan_raw_tasks` and `load_raw_task`.
- `config.py`: define `DatasetSpec`, ingest/normalize specs, and `DATASET`.
- `registry.py`: register the dataset id so `get_dataset` can find it.

Raw adapters yield `IngestBatch` objects. Batch metadata must include the protocol
and the protocol task keys used for manifests and normalization tasks. If a raw
source omits optional declared columns, add null columns in the adapter before
yielding the batch; generic ingest alignment treats missing declared columns as
errors. The Pozzato adapter is the current example: it infers protocol/task
metadata from file paths, requires exactly one matching Excel sheet, and fills
missing declared raw columns with nulls before alignment.

### Run Ingest

```python
from pathlib import Path

from batgrad.data.datasets.registry import get_dataset
from batgrad.data.processing.raw import IngestStageConfig
from batgrad.storage.local import LocalDataProcessingStore

dataset = get_dataset("pozzato-2022")
store = LocalDataProcessingStore(Path("/data/batgrad"), create=True)

dataset.ingest(
    input_store=store,
    output_store=store,
    config=IngestStageConfig(n_jobs=-1),
)
```

This writes ingested parquet shards and an ingested manifest.

### Run Normalize

```python
from batgrad.data.processing.normalize import NormalizeStageConfig

dataset.normalize(
    input_store=store,
    output_store=store,
    config=NormalizeStageConfig(n_jobs=-1),
)
```

This reads the ingested manifest and writes normalized parquet shards plus a
normalized manifest. Use `dry_run=True` to validate tasks without writing outputs;
use `max_batch_rows` to keep large tasks on the bounded processing path. Bounded
normalization requires bounded-capable resampling; for protocols using linear
resampling, keep tasks on the full-task path with `max_batch_rows=None` or a
large enough limit.

### Interactive Runs

```python
run = dataset.normalize_interactive(
    input_store=store,
    scratch_store=store,
    config=NormalizeStageConfig(n_jobs=1),
    protocols="cycling",
    group_values={"cell id": "A1"},
)

manifest = run.manifest()
data = run.scan()
run.clean()
```

`protocols` filters protocols, `group_values` filters task keys such as cell or
cycle, and `scratch_store` receives temporary outputs.

## Details

Use the API Reference for implementation details:

- API Reference > Data
- API Reference > Contracts
- API Reference > Datastore
