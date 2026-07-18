# Quick Start

## Environment

Create the environment files:

```sh
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env
```

Before starting Compose, replace `HOST_DATA_ROOT` in `.env` and optionally configure trusted
dotfiles. File mode `600` is optional but recommended when either env file contains secrets. Then
start the local dev container and run project setup:

```sh
docker compose up -d --build dev
docker compose exec -it dev zsh
./scripts/setup_project.sh
```

For remote containers, provider images, data mounts, and environment variables,
see [Environment Setup](environment-setup.md).

## Explore the Notebooks

Download the released normalized datasets to `DATA_ROOT` and the inference checkpoint to
`outputs/checkpoints/`:

```sh
uv run scripts/hf_assets.py download
```

Start the local Marimo editor:

```sh
uv run marimo edit notebooks \
  --headless \
  --host 0.0.0.0 \
  --port 2718 \
  --session-ttl 5
```

Open `http://localhost:2718`. On a remote instance, use the SSH tunnel from
[Environment Setup](environment-setup.md) and open the same local URL.

The five notebook entrypoints follow the workflow from data to evaluation:

| Notebook | Use |
| --- | --- |
| `etl.py` | Inspect ingestion, transformations, resampling, and normalized data |
| `dataloader.py` | Inspect manifests, splits, temporal windows, and batches |
| `config.py` | Build and validate experiment configurations |
| `training.py` | Step through optimization and validation traces |
| `inference.py` | Compare checkpoints and multi-step rollouts |

The released Mamba-3 checkpoint in `inference.py` requires CUDA. These notebooks are intended for
interactive exploration rather than unattended long-running jobs.

## Data Processing

The data flow is `raw files -> ingested parquet -> normalized parquet`. Interactive
runs write scratch outputs for inspection without replacing persisted stage data.
The public Pozzato source supports this full path. The released synthetic asset is
normalized data; its raw Parquet inputs and generation pipeline are not public here.
For large runs, place `scratch_store` on persistent storage with enough capacity for
concurrent task outputs rather than relying on container `/tmp`.

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
store = LocalDataProcessingStore(Path("/data"), create=True)

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

## Machine Learning

The ML layer reads normalized manifests. Complete ingestion and normalization or download the
released assets above, then choose a configuration whose manifest revision and selected columns
match the normalized data.

The bundled configurations provide useful starting points:

- `configs/ml_dry_run_cpu.json` runs a short
  attention/FFN smoke test on CPU without writing run artifacts.
- `configs/ml_dry_run_gpu.json` runs a short Mamba smoke
  test on CUDA without writing run artifacts.
- `configs/ml_baseline.json` is the full CUDA/W&B baseline
  and writes logs and checkpoints.

Use `notebooks/config.py` to edit the full schema. Its scaling editor derives
one explicit rule per selected input or target column. Known battery columns start
with editable suggestions, but the saved JSON always contains the resolved numeric
rules and should be reviewed against normalized manifest statistics.

Set `data.store_root` in the selected configuration or ensure `DATA_ROOT` points
to the data store, then run:

```sh
uv run scripts/train.py --config configs/ml_dry_run_cpu.json
uv run scripts/train.py --config configs/ml_dry_run_gpu.json
uv run scripts/train.py --config configs/ml_baseline.json
```

The baseline logs online to the `batgrad` W&B project. Authenticate with `uv run wandb login`, or
select stdout, JSONL, or offline W&B in the configuration. Run the baseline on two GPUs with DDP:

```sh
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/ml_baseline.json
```

For file-backed runs, `train_from_config` creates the configured
`run.output_dir/run.name` directory. If `run.name` is omitted, it uses a local
timestamp. A named directory that already exists is deleted before the run
starts.

```text
<run-dir>/
  config.json
  logs/
    metrics.jsonl
    payloads.jsonl
  checkpoints/
    <run-id-or-name>/
      latest.pt
      best_<metric>.pt
      final.pt
```

Only enabled logger and checkpoint outputs are created. See
[ML Configuration](api/ml/configuration.md),
[ML Data Loading](api/ml/data-loading.md), [ML Models](api/ml/models.md),
[ML Training](api/ml/training.md), and [ML Inference](api/ml/inference.md).
