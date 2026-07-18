# batgrad

`batgrad` is an opinionated, hackable template for battery time-series foundation models.

It provides:

- Parquet datasets for consistent schemas and efficient I/O.
- Per-stage manifests indexing Parquet shards, stream metadata, and provenance.
- A local-first workflow designed for fast iteration within CPU, GPU, and memory constraints.
- Revision-pinned manifests and saved experiment configurations for traceable, reproducible runs.
- Interactive [Marimo](https://marimo.io/) notebooks for exploring, demonstrating, and debugging the pipeline. They are a playground, not a production UI or experiment runner.

## Quickstart

Run the hosted notebooks on [Molab](https://molab.marimo.io/<tbd>). Molab is Marimo's hosted notebook environment, similar to Colab but backed by reactive Python files rather than order-dependent cells.

Normalized datasets are provided through the
[batgrad dataset repository](https://huggingface.co/datasets/marplan6/batgrad), with example
checkpoints in the [batgrad model repository](https://huggingface.co/marplan6/batgrad).

Hosted notebooks:

- [Data processing](https://molab.marimo.io/<tbd>)
- [Training](https://molab.marimo.io/<tbd>)
- [Inference](https://molab.marimo.io/<tbd>)

## Development

Using the provided Docker image as a development container is strongly recommended. The environment targets Linux; Windows is unsupported and macOS is untested. The container runs as the non-root `ubuntu` user.

The baseline architecture uses Mamba and requires a CUDA-capable GPU (~10M params, ~4GB VRAM inference). Data processing is CPU-only, and `configs/ml_dry_run_cpu.json` provides a CPU-compatible training configuration without Mamba layers.

Container layout:

```text
/workspace/ubuntu/batgrad/           # project
/workspace/ubuntu/batgrad/outputs/   # runs and checkpoints
/data/                               # DATA_ROOT
```

### Local Setup

Local setup requires Docker with Docker Compose. A CUDA-capable GPU is required to run the baseline model; it has been tested on Blackwell and is expected to support Ampere and Hopper.

```sh
git clone https://github.com/marplan/batgrad.git
cd batgrad
cp .env.example .env
```

Set `HOST_DATA_ROOT` in `.env` to the host directory mounted at `/data` in the container:

```text
~/my_projects/
├── batgrad/
└── my_data/       # HOST_DATA_ROOT, mounted at /data
```

Personal dotfiles are optional:

```sh
cp docker/dotfiles.env.example docker/dotfiles.env
```

Configure `docker/dotfiles.env` if used. Additional system tools can be added to `docker/Dockerfile` or `docker/brew-packages.txt` before building.

Build and enter the development container, then set up the project:

```sh
docker compose up -d dev --build
docker compose exec dev zsh
./scripts/setup_project.sh
```

### Remote Setup

The full environment can run on [Vast.ai](https://cloud.vast.ai?ref_id=140701&template_id=122905af35856f2e6e6ab6d925bebd06) or [RunPod](https://console.runpod.io/deploy?template=7npkwu4zr5&ref=7z0qtxil) using the published `batgrad` template.

The image provides direct access as the non-root `ubuntu` user. Replace the provider's default `root@<host>` with `ubuntu@<host>`. Agent forwarding with `-A` is recommended, and `-L 2718:localhost:2718` exposes Marimo locally.

```sh
ssh -A -p <port> ubuntu@<host> -L 2718:localhost:2718
```

On the remote host:

```sh
cd /workspace/ubuntu
git clone git@github.com:marplan/batgrad.git
cd batgrad
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env  # optional
```

Configure `docker/dotfiles.env` if used, then run:

```sh
./scripts/setup_project.sh
```

### Explore

Download both normalized baseline datasets to `DATA_ROOT`:

```sh
uv run scripts/hf_assets.py download
```

Download the example checkpoint to `outputs/checkpoints/` when available:

```sh
uv run scripts/hf_assets.py download --ckpt batgrad_init_baseline
```

Start the Marimo development server:

```sh
uv run marimo edit notebooks \
  --headless \
  --host 0.0.0.0 \
  --port 2718 \
  --session-ttl 5
```

On a remote host, forward port `2718` to your local machine. `--session-ttl` terminates notebook sessions after the browser disconnects.

## How It Works

`batgrad` separates dataset-specific ingestion from reusable data processing and ML:

```text
raw sources
  -> dataset adapter and canonical mappings
  -> ingested Parquet + manifest
  -> transforms, checks, and resampling
  -> normalized Parquet + manifest
  -> ML index, loaders, training, and inference
```

The main directories reflect these boundaries:

```text
batgrad/contracts/       Shared columns, protocols, and metadata contracts
batgrad/data/            Dataset adapters and processing stages
batgrad/ml/              Data loading, models, training, and inference
configs/                 Reproducible experiment configurations
notebooks/               Interactive exploration and debugging
scripts/                 Data-processing and training entry points
```

See the [documentation](docs/index.md) for a more complete reference.

### Datasets

Published battery datasets commonly mix Excel, CSV, Parquet, and dataset-specific naming conventions. `batgrad` keeps those irregularities inside each dataset adapter and converts them into canonical, protocol-sharded Parquet data.

The persisted stages are:

```text
raw -> ingested -> normalized
```

- **Ingested** data uses canonical column names and dtypes while retaining source paths and task metadata.
- **Normalized** data applies configured transformations, checks, and resampling for analysis and ML.
- Each processed stage includes a `manifest.parquet` describing its streams, exact shard segments, metadata, and Git provenance.

#### Included Datasets

- **`pozzato-2022`**: the published NMC/graphite battery-aging dataset by Pozzato, Allam, and Onori, with cycling, HPPC, RPT, and EIS data.
- **`synthetic-pozzato-2022-m50t`**: a pre-generated synthetic LG INR21700 M50T dataset associated with `pozzato-2022`.

Pozzato 2022 resources:

- [Dataset overview](https://osf.io/qsabn/overview?view_only=2a03b6c78ef14922a3e244f3d549de78)
- [Raw dataset download](https://www.dropbox.com/scl/fo/3ss0age6ggfcm67okldhw/h?rlkey=tnczvb82gukfe2n4gol2uyo7x&dl=0)
- [Publication](https://doi.org/10.1016/j.dib.2022.107995)
- License: CC BY 4.0

Approximate Pozzato 2022 footprint with the current processing configuration:

| Stage              |    Size |                Processing |
| ------------------ | ------: | ------------------------: |
| Raw Excel          | ~300 GB |                         - |
| Ingested Parquet   |  ~60 GB |    ~1 hour with 3 workers |
| Normalized Parquet |   ~2 GB | ~3 minutes with 3 workers |

Peak memory remains below approximately 3 GB per worker with the default settings. These figures are indicative and depend on the machine and processing configuration.

#### Process Datasets

Place the raw files under the dataset's raw stage, for example:

```text
/data/type=published/dataset=pozzato-2022/source=raw/
```

Run ingestion and normalization from the command line:

```sh
uv run scripts/data_processing.py --ingest pozzato-2022
uv run scripts/data_processing.py --normalize pozzato-2022
```

Both stages can be run in one invocation:

```sh
uv run scripts/data_processing.py \
  --ingest pozzato-2022 \
  --normalize pozzato-2022 \
  --n-jobs 3
```

`--ingest` and `--normalize` accept one or more registered dataset IDs or `all`. When both are supplied, all ingestion jobs complete before normalization begins. `--store` selects the data root, `--scratch-store` selects temporary storage, and `--n-jobs` controls stage parallelism.

The same pipeline is available from Python:

```python
from batgrad.data.datasets.registry import get_dataset
from batgrad.data.processing.normalize import NormalizeStageConfig
from batgrad.data.processing.raw import IngestStageConfig
from batgrad.storage.local import LocalDataProcessingStore

dataset = get_dataset("pozzato-2022")
store = LocalDataProcessingStore("/data")

dataset.ingest(
    input_store=store,
    output_store=store,
    config=IngestStageConfig(n_jobs=3),
)

dataset.normalize(
    input_store=store,
    output_store=store,
    config=NormalizeStageConfig(n_jobs=3),
)
```

The registered dataset supplies its raw adapter and stage definitions. The runtime configurations control execution and Parquet-writing behavior.

#### Add a Dataset

A dataset implementation keeps source-specific behavior in four files:

```text
batgrad/data/datasets/<dataset_id>/
├── __init__.py
├── mapping.py
├── raw.py
└── config.py
```

- `mapping.py` maps source columns and aliases to canonical contracts.
- `raw.py` discovers source files and yields ingestion batches.
- `config.py` defines protocols, columns, transformations, checks, resampling, and dataset metadata.
- `registry.py` exposes the dataset to the CLI and Python API.

See [Adding a dataset](docs/quick-start.md#add-a-dataset) for the complete adapter contract.

### Machine Learning

The ML pipeline consumes normalized manifests rather than raw files. It validates their schemas and expected Git revisions, combines selected manifest rows into an in-memory index, creates group-aware train and validation streams, and materializes fixed-length input and target windows.

Each experiment is defined by one strict JSON configuration covering:

- Manifests, expected producer revisions, and protocols
- Input, target, feedback, and scaling columns
- Loader and state-carry behavior
- Attention, FFN, and optional Mamba layers
- Optimization, validation, logging, and checkpoints

The validated configuration is saved with file-backed runs. Logging supports stdout, JSONL, and W&B, while checkpoints contain the model and training state together with the complete experiment configuration.

Run the bundled configurations with:

```sh
# Short attention/FFN smoke test on CPU
uv run scripts/train.py --config configs/ml_dry_run_cpu.json

# Short Mamba smoke test on CUDA
uv run scripts/train.py --config configs/ml_dry_run_gpu.json

# Full CUDA/W&B baseline
uv run scripts/train.py --config configs/ml_baseline.json
```

The baseline logs online to the `batgrad` W&B project under the logged-in account. Authenticate once with `uv run wandb login`, or edit `logging` in the JSON config to use a different project, group, mode, or backend.

Run the baseline on two GPUs with DDP:

```sh
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/ml_baseline.json
```

Use `notebooks/config.py` to create and validate configurations interactively. The notebooks are intended for exploration and debugging; `scripts/train.py` is the experiment entry point.

See the [ML documentation](docs/index.md#ml-flow) for configuration, data loading, models, training, and inference details.
