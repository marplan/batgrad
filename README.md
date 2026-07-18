<h1 align="center">batgrad</h1>

<p align="center">
  <a href="https://marplan.github.io/batgrad/">Docs</a>
</p>

<p align="center">
  <strong>An opinionated, hackable template for battery time-series models.</strong>
</p>

`batgrad` turns experimental and synthetic battery data into traceable datasets for configurable
neural-network training and multi-step evaluation.

## Why batgrad?

- **Normalize messy data once.** Dataset adapters handle source files, columns, and protocol
  metadata; the shared pipeline handles the rest.
- **Process and mix large datasets.** Sharded Parquet keeps sources separate and traceable while
  bounded-capable transforms keep memory use practical.
- **Split by battery structure.** Validation groups preserve dataset, cell, cycle, and protocol
  boundaries instead of sampling nearby rows.
- **Train and inspect multi-step models.** Attention, FFN, and Mamba-3 layers predict future chunks;
  five Marimo notebooks expose the complete workflow.

**Baseline model:** approximately 10M parameters.

Released [normalized datasets](https://huggingface.co/datasets/marplan6/batgrad) and the
[inference checkpoint](https://huggingface.co/marplan6/batgrad) are available on Hugging Face.

## Explore in Molab

[Molab](https://molab.marimo.io/) runs the same Marimo notebooks in the browser.

> [!NOTE]
>
> - Molab uses `/marimo/batgrad` as an editable checkout and `/marimo/data` for datasets.
> - Initial setup can take a few minutes for training and inference. Check the setup-cell output;
>   the first training and validation step can also take some time.
> - The notebooks are for interactive discovery, not long-running jobs.
> - Molab downloads normalized data only. Raw ETL requires source files and the
>   [local data-processing workflow](https://marplan.github.io/batgrad/quick-start/#data-processing).

| Notebook      | Purpose                                                  | Launch                                                                                                                                   |
| ------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| ETL           | Inspect stages, transforms, resampling, and source rows  | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/etl.py)        |
| Data loader   | Inspect manifests, splits, temporal windows, and batches | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/dataloader.py) |
| Configuration | Build and validate experiment configurations             | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/config.py)     |
| Training      | Step through real optimization and validation traces     | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/training.py)   |
| Inference     | Compare checkpoints and multi-step rollouts              | [![Open in Molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/marplan/batgrad/blob/main/notebooks/inference.py)  |

Select a CUDA/GPU Molab runtime for the released Mamba-3 inference checkpoint; CPU runtimes support
only non-Mamba checkpoints.

## Quickstart

> [!IMPORTANT]
> **Requirements:** Docker Compose on Linux; about 7.2 GiB for released data (raw ETL needs more),
> about 4 GB RAM per processing worker, and about 4 GB VRAM with CUDA 13 for baseline Mamba
> inference. A W&B account is optional and needed only for online logging.

### Local Docker

Clone the repository and configure the host directory mounted at `/data`:

```sh
git clone https://github.com/marplan/batgrad.git
cd batgrad
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env
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

Open `http://localhost:2718`.

### Remote GPU Development

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

Keep any provider-supplied `-i` and identity options. Omit `-A` unless a trusted instance needs
your local SSH agent for private dotfiles or dependencies. After connecting, prepare the checkout:

```sh
cd /workspace/ubuntu
git clone https://github.com/marplan/batgrad.git
cd batgrad
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env
```

Review `.env`. If wanted, enable and configure trusted personal dotfiles in
`docker/dotfiles.env`. File mode `600` is optional but recommended if either file contains secrets.
Then run setup:

```sh
./scripts/setup_project.sh
```

Continue with the asset-download and Marimo commands in the local workflow above.

See [Environment Setup](https://marplan.github.io/batgrad/environment-setup/) for provider setup,
paths, dotfiles, and image customization.

## How It Works

```text
messy source files
  -> dataset adapter: discovery, mappings, parsers, metadata
  -> canonical ingested Parquet + manifest
  -> shared transforms, checks, resampling, and sharding
  -> normalized Parquet + manifest
  -> notebooks, ML loaders, training, and inference
```

Dataset adapters translate source-specific files and metadata into canonical ingested Parquet.
Normalization adds checks, transformations, derived features, and protocol-specific resampling.
Both stages use compressed, protocol-sharded Parquet and manifests that retain exact row segments,
source paths, stream metadata, dataset identity, and the producer Git revision. ML loading rejects
unexpected revisions before building windows; revisions identify producer code, not raw-file
content.

## Datasets

### Included Datasets

| Dataset ID                    | Conditions                                                   | Protocols               | Availability       |
| ----------------------------- | ------------------------------------------------------------ | ----------------------- | ------------------ |
| `pozzato-2022`                | CC-CV charging and dynamic discharge cycling at 23 °C        | Cycling, HPPC, RPT, EIS | Raw and normalized |
| `synthetic-pozzato-2022-m50t` | 10–30 °C, five cooling settings, four dynamic profile scales | Cycling, RPT, EIS       | Normalized only    |

Pozzato's source chamber was 23 °C; its normalized ambient and cooling controls are fixed to 20 °C
and 20 W·m⁻²·K⁻¹. The synthetic release carries its varied thermal conditions as inputs.

Pozzato 2022: [overview](https://osf.io/qsabn/overview?view_only=2a03b6c78ef14922a3e244f3d549de78),
[raw files](https://www.dropbox.com/scl/fo/3ss0age6ggfcm67okldhw/h?rlkey=tnczvb82gukfe2n4gol2uyo7x&dl=0),
and [publication](https://doi.org/10.1016/j.dib.2022.107995). The source and processed derivative use
CC BY 4.0.

Approximate Pozzato 2022 footprint with three workers; results vary by hardware and configuration:

| Stage              |    Size | Processing |
| ------------------ | ------: | ---------: |
| Raw Excel          | ~300 GB |          - |
| Ingested Parquet   |  ~60 GB |    ~1 hour |
| Normalized Parquet |   ~2 GB | ~3 minutes |

### Process Raw Data

Place the published raw files under the canonical raw root:

```text
/data/type=published/dataset=pozzato-2022/source=raw/
```

Run the published dataset end to end, keeping temporary shards on persistent storage:

```sh
uv run scripts/data_processing.py \
  --ingest pozzato-2022 \
  --normalize pozzato-2022 \
  --scratch-store /data/scratch \
  --n-jobs 3
```

`--store` sets the data root, `--scratch-store` sets temporary storage, and `--n-jobs` controls
parallelism. Size scratch storage for concurrent outputs. Use `all` for every registered dataset
with available raw sources; the public raw-to-normalized workflow currently applies to Pozzato.

### Add a Dataset

A dataset adapter defines discovery and loading in `raw.py`, canonical mappings in `mapping.py`,
and protocol processing in `config.py`. Register it to use the shared ETL, manifests, notebooks,
and ML loaders. See [Adding a dataset](https://marplan.github.io/batgrad/quick-start/#add-a-dataset).

## Machine Learning

ML loaders read normalized manifests, combine compatible datasets without copying them, split by
battery metadata, and build one-row-ahead windows. The approximately 10M-parameter baseline mixes
categorical feature encodings with configurable Attention, FFN, and Mamba-3 layers to predict a
future chunk in parallel.

The baseline sees time difference, current, voltage, surface temperature, ambient temperature, and
cooling. During rollout, time difference, current, ambient temperature, and cooling remain known
controls; voltage and surface temperature are predicted and fed back. This is why learning across
different current and thermal conditions matters. Training uses one window, while evaluation can
extend it through recursive rollout chunks.

One validated JSON file defines data revisions, features, scaling, loaders, model, optimization,
validation, logging, and checkpoints. It is saved with file-backed runs and checkpoints.

Run the bundled configurations:

```sh
uv run scripts/train.py --config configs/ml_dry_run_cpu.json
uv run scripts/train.py --config configs/ml_dry_run_gpu.json
uv run scripts/train.py --config configs/ml_baseline.json
```

The baseline logs online to the `batgrad` W&B project. Run `uv run wandb login`, or select stdout,
JSONL, or offline W&B; those modes need no account.

Run the baseline on two GPUs with DDP:

```sh
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc-per-node=2 \
  scripts/train.py --config configs/ml_baseline.json
```

Full checkpoints contain training state and configuration. The Hugging Face checkpoint contains
weights, configuration, step, and format version. The trainer can initialize from weights but cannot
yet resume an interrupted run.

See the [ML documentation](https://marplan.github.io/batgrad/#ml-flow) for data loading,
architecture, training, validation, and inference.

## Notes and Limitations

- EIS is a processing example and is not used by the bundled ML workflows.
- The synthetic release contains normalized data only; its generator and raw
  files are not provided.
- This is one reference workflow, not the sweeps and broader experiments used
  to select it. Logging is meant for inspecting individual runs, not comparing
  a research campaign.
- Validation uses held-out rows from the selected datasets. It does not show
  transfer to unseen datasets or operating conditions.
- AI helped with implementation, tests, and documentation. I directed nearly
  every change step by step, reviewed most of the code, and wrote parts by hand
  rather than generating the project in one prompt.
- Tested on RTX 5090/Blackwell. Windows is unsupported. macOS is untested; CPU
  and data paths may work, but CUDA, Mamba, and Linux-only tools do not.
- The development image uses Linuxbrew; production environments may prefer a
  native toolchain.
