# ML Inference

Use `rollout_batch` when a compatible model and tensors are already available.
Use `evaluate_checkpoints` to materialize selected normalized streams once and
compare several checkpoints and rollout suffix widths.

The comparative entry point accepts rows from an ML index:

```python
import polars as pl
import torch

from batgrad.contracts.mapping import BaseColumns
from batgrad.ml.config import load_experiment_config, resolve_store_root
from batgrad.ml.data.loader import create_index
from batgrad.ml.inference import CheckpointSelection, evaluate_checkpoints
from batgrad.storage.local import LocalDataProcessingStore

config = load_experiment_config("configs/ml_baseline.json")
store = LocalDataProcessingStore(resolve_store_root(config.data.store_root))
index = create_index(
    store,
    config.data.manifest_paths,
    protocols=config.data.protocols,
    protocol_mode=config.data.protocol_mode,
)
selected = index.frame.filter(pl.col(BaseColumns.proto) == "cycling").head(1)

result = evaluate_checkpoints(
    store,
    selected,
    (CheckpointSelection("baseline", "output/run/checkpoints/run/final.pt"),),
    device=torch.device("cuda:0"),
    suffix_steps=(0, config.train.masked_suffix.suffix_steps),
    rollout_steps=256,
)
if result.warning:
    print(result.warning)
```

Suffix width `0` selects classic one-step rollout. Positive widths use masked
suffix rollout and must be smaller than the checkpoint context length. Returned
predictions are decoded in the configured output-scaled units; inverse-scale them
before comparing with physical measurements. Comparative inference starts at
source-row offset zero for every selected stream and does not currently support
EIS.

::: batgrad.ml.rollout.rollout_batch
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.rollout.RolloutResult
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.inference.evaluate_checkpoints
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.inference.resolve_device
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.inference.available_devices
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.inference.CheckpointSelection
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.inference.InferencePrediction
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.inference.InferenceResult
    options:
      heading_level: 2
      members: false
      show_source: false

## Checkpoints

::: batgrad.ml.inference.discover_checkpoints
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.checkpoint.read_checkpoint_config
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.checkpoint.load_checkpoint
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.checkpoint.LoadedCheckpoint
    options:
      heading_level: 3
      members: false
      show_source: false

## Metrics

::: batgrad.ml.metrics.LossMetrics
    options:
      heading_level: 3
      members: false
      show_source: false
