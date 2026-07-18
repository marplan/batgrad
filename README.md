<h1 align="center">batgrad</h1>

<p align="center">
  <strong>An opinionated, hackable template for battery time-series models.</strong>
</p>

`batgrad` brings heterogeneous experimental and synthetic battery data into one pipeline: from
messy raw files, through canonical and traceable datasets, to configurable neural-network training
and multi-step evaluation.

## Why batgrad?

- **Onboard messy datasets once.** A small adapter handles source files, column mappings, and
  protocol metadata; shared ingestion, normalization, notebook, and ML pipelines take over from
  there.
- **Process large datasets locally.** Bounded-memory normalization, protocol-level parallelism,
  and sharded Parquet keep hundreds of gigabytes manageable on ordinary CPU machines.
- **Mix datasets without losing identity.** Published and synthetic sources share canonical
  columns while retaining dataset, cell, cycle, protocol, source, and revision provenance.
- **Split by battery structure, not random rows.** Training and validation groups preserve
  dataset, cell, cycle, and protocol boundaries to reduce leakage.
- **Model more than the next sample.** The hybrid Attention-FFN-Mamba-3 model can predict several
  future steps in parallel and validate behavior through recursive rollouts.
- **Inspect the real pipeline.** Five reactive Marimo workbenches expose ETL, manifests, loaders,
  configuration, training, and inference.

**Compact baseline:** approximately 10M parameters and 4 GB VRAM for inference on a single
CUDA-capable GPU.

Normalized datasets are available from the
[batgrad dataset repository](https://huggingface.co/datasets/marplan6/batgrad), with a compact
checkpoint in the [batgrad model repository](https://huggingface.co/marplan6/batgrad).

## Explore in Molab

[Molab](https://molab.marimo.io/) runs Marimo notebooks in the browser. Each notebook below is a
reactive Python file backed by the same library code used by the command-line workflows.

| Notebook | Purpose | Launch |
| --- | --- | --- |
| ETL | Inspect stages, transforms, resampling, and source rows | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/etl.py) |
| Data loader | Inspect manifests, splits, temporal windows, and batches | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/dataloader.py) |
| Configuration | Build and validate experiment configurations | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/config.py) |
| Training | Step through real optimization and validation traces | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/training.py) |
| Inference | Compare checkpoints and multi-step rollouts | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/inference.py) |

## Quickstart

> [!IMPORTANT]
> **Requirements:** Docker with Docker Compose and a Linux environment. Data processing is
> CPU-only and uses approximately 3 GB RAM per worker for the included Pozzato pipeline. Mamba-3
> requires Linux and CUDA; one CUDA GPU is sufficient for the baseline workflow. W&B is only
> required for online logging.

Clone the repository and configure the host directory mounted at `/data`:

```sh
git clone https://github.com/marplan/batgrad.git
cd batgrad
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env  # optional
```

Set `HOST_DATA_ROOT` in `.env`. If used, configure `docker/dotfiles.env` before building the
non-root development container and installing the project:

```sh
docker compose up -d dev --build
docker compose exec dev zsh
./scripts/setup_project.sh
```

Download all released datasets to `DATA_ROOT` and checkpoints to `outputs/checkpoints/`, then
start Marimo:

```sh
uv run scripts/hf_assets.py download
uv run marimo edit notebooks \
  --headless \
  --host 0.0.0.0 \
  --port 2718 \
  --session-ttl 5
```

Open `http://localhost:2718`. Personal dotfiles and additional container tools are documented in
the [environment setup guide](docs/environment-setup.md).

## How It Works

```text
messy source files
  -> dataset adapter: discovery, mappings, parsers, metadata
  -> canonical ingested Parquet + manifest
  -> shared transforms, checks, resampling, and sharding
  -> normalized Parquet + manifest
  -> notebooks, ML loaders, training, and inference
```

Manifests are the handoff between stages. They index exact row segments in protocol-sharded
Parquet files and retain source paths, stream metadata, dataset identity, and producer Git
provenance. Experiment configurations declare expected manifest revisions, and ML loading rejects
revision mismatches before constructing windows.

This is a traceability guardrail rather than immutable data versioning: revisions identify the
producer code, not the content of every source file.

## Data Pipeline

Published battery datasets commonly mix Excel, CSV, Parquet, inconsistent column names, and
source-specific layouts. A dataset adapter contains those irregularities at the edge of the
system. Once data reaches the ingested stage, downstream processing no longer needs to know how
the source was packaged.

The persisted stages are:

```text
raw -> ingested -> normalized
```

- **Ingested** data has canonical columns and dtypes while retaining source paths and task
  metadata.
- **Normalized** data applies configured transformations, checks, derived features, and
  protocol-specific resampling for analysis and ML.
- Both stages write compressed Parquet with content-defined chunking and manifests with exact
  segment references.

### Included Datasets

| Dataset ID | Source | Protocols |
| --- | --- | --- |
| `pozzato-2022` | Published NMC/graphite battery-aging measurements | Cycling, HPPC, RPT, EIS |
| `synthetic-pozzato-2022-m50t` | PyBaMM-generated LG INR21700 M50T profiles | Cycling, RPT, EIS |

Pozzato 2022 resources: [dataset overview](https://osf.io/qsabn/overview?view_only=2a03b6c78ef14922a3e244f3d549de78),
[raw data](https://www.dropbox.com/scl/fo/3ss0age6ggfcm67okldhw/h?rlkey=tnczvb82gukfe2n4gol2uyo7x&dl=0),
and [publication](https://doi.org/10.1016/j.dib.2022.107995). The source and processed derivative
are licensed under CC BY 4.0.

Indicative Pozzato 2022 footprint with three workers:

| Stage | Size | Processing |
| --- | ---: | ---: |
| Raw Excel | ~300 GB | - |
| Ingested Parquet | ~60 GB | ~1 hour |
| Normalized Parquet | ~2 GB | ~3 minutes |

Actual time, memory, and storage depend on hardware and processing configuration.

### Process Raw Data

Place raw files under each dataset's canonical raw root, for example:

```text
/data/type=published/dataset=pozzato-2022/source=raw/
/data/type=synthetic/dataset=synthetic-pozzato-2022-m50t/source=raw/
```

> [!WARNING]
> Ingestion and normalization replace the selected dataset stage output. Use a separate store or
> preserve existing outputs before rerunning a stage.

Run both datasets end to end:

```sh
uv run scripts/data_processing.py \
  --ingest pozzato-2022 synthetic-pozzato-2022-m50t \
  --normalize pozzato-2022 synthetic-pozzato-2022-m50t \
  --n-jobs 3
```

`--store` selects the data root, `--scratch-store` selects temporary storage, and `--n-jobs`
controls stage parallelism. Use `all` instead of explicit dataset IDs to process every registered
dataset.

### Add a Dataset

A new source implements its file discovery and loading in `raw.py`, canonical mappings in
`mapping.py`, and protocol processing policy in `config.py`. Register it once and the shared ETL,
manifest, notebook, and ML workflows become available. See
[Adding a dataset](docs/quick-start.md#add-a-dataset) for the adapter contract.

## Machine Learning

The ML pipeline consumes normalized manifests rather than raw files. It combines compatible
datasets without physically merging them, forms train and validation groups from battery metadata,
and materializes one-row-ahead temporal windows directly from manifest segments.

The baseline encodes continuous inputs into bounded categorical representations, projects each
feature independently, and mixes them through configurable Attention, FFN, and Mamba-3 layers.
Outputs are categorical distributions over each target. Future feedback channels can be hidden so
the model predicts several future steps from one context; recursive validation then feeds
predictions back into the model to test longer closed-loop rollouts with optional known future
controls.

One strict JSON configuration defines data revisions, protocols, columns, scaling, loader behavior,
model structure, optimization, validation, logging, and checkpoints. The validated configuration is
saved with file-backed runs and embedded in checkpoints.

Run the bundled configurations:

```sh
# Short Attention/FFN smoke test on CPU
uv run scripts/train.py --config configs/ml_dry_run_cpu.json

# Short Mamba-3 smoke test on CUDA
uv run scripts/train.py --config configs/ml_dry_run_gpu.json

# Full CUDA/W&B baseline
uv run scripts/train.py --config configs/ml_baseline.json
```

The baseline logs online to the `batgrad` W&B project. Authenticate with `uv run wandb login`, or
change the logging backend or mode in the configuration.

Run the baseline on two GPUs with DDP:

```sh
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/ml_baseline.json
```

> [!WARNING]
> Reusing an explicit `run.name` replaces its existing run directory. `run.init_from` initializes
> compatible model weights only; it does not resume optimizer, scheduler, scaler, or training
> cursor state.

Full run checkpoints contain training state and the complete configuration. The compact Hugging
Face checkpoint contains model weights, configuration, step, and format version for inference and
weight initialization.

See the [ML documentation](docs/index.md#ml-flow) for the architecture, data loading, training,
validation, and inference contracts.

## Run Locally or Remotely

The non-root Docker environment is shared across local development,
[Vast.ai](https://cloud.vast.ai?ref_id=140701&template_id=122905af35856f2e6e6ab6d925bebd06)
and [RunPod](https://console.runpod.io/deploy?template=7npkwu4zr5&ref=7z0qtxil). Remote development
uses the same local-filesystem workflow on storage attached to the GPU host.

> [!IMPORTANT]
> Vast.ai and RunPod commonly provide an SSH command using `root@<host>`. The batgrad image uses
> the non-root `ubuntu` user instead. Keep the provider's port and identity options, but replace
> `root@<host>` with `ubuntu@<host>`.

Forward SSH credentials and the Marimo port when connecting:

```sh
ssh -A -L 2718:localhost:2718 -p <port> ubuntu@<host>
```

See the [environment setup guide](docs/environment-setup.md) for provider setup, paths, dotfiles,
and image customization.

## Deliberate Scope

- EIS is treated as a data-processing example in this template; it is not part of the bundled
  training or inference workflows.
- The released synthetic dataset is included, but its PyBaMM generation pipeline is intentionally
  omitted to keep the codebase focused.
- Hyperparameter sweeps and the broader training and validation experiments used to select the
  baseline architecture are not included; the repository keeps one compact reference workflow.
- The development image uses Linuxbrew for convenience, not as a production packaging
  recommendation. Production deployments should derive a pinned, hardened image with their
  preferred system-package strategy.
- Blackwell is tested; Ampere and Hopper are expected but currently unverified. Windows is
  unsupported and macOS is untested.
