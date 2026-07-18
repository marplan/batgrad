# ML Training

Training is configuration-driven. The entry point owns data loading, model and
optimizer construction, validation, logging, distributed setup, and checkpoint
persistence.

Run a single process with:

```sh
python scripts/train.py --config configs/ml_dry_run_cpu.json
```

Distributed training is enabled by the environment that `torchrun` creates and
currently requires CUDA and NCCL:

```sh
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --config configs/ml_baseline.json
```

File-backed runs create the following structure. W&B runs use the W&B run ID for
the nested checkpoint directory; other loggers use the local run name.

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

Files are conditional on the selected logger and checkpoint flags. Best
checkpoints minimize their configured metric. `run.init_from` initializes
compatible model weights only; it does not restore optimizer, scheduler, AMP
scaler, or training cursor state.

`latest` and `best_<metric>` are updated only when validation runs. A missing or
misspelled monitor emits a warning and does not produce a best checkpoint. Common
monitor names include `val/tf/loss_ce`, `val/tf/rmse`,
`val/rollout/loss_ce`, and `val/rollout/rmse`.

**Warning:** If `run.name` resolves to an existing directory under
`run.output_dir`, the directory is deleted before training starts.

::: batgrad.ml.train.train_from_config
    options:
      heading_level: 2
      show_source: false
