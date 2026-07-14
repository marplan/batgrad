# ML Configuration

The experiment configuration is a strict, frozen dataclass contract. Unknown
JSON fields and incompatible settings are rejected before data loading or model
execution begins. Mapping-valued fields remain ordinary dictionaries and should
be treated as read-only. Start from the bundled `configs/ml_dry_run_cpu.json`,
`configs/ml_dry_run_gpu.json`, or `configs/ml_baseline.json` rather than
constructing the full schema from memory.

## Loading

::: batgrad.ml.config.load_experiment_config
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.config.parse_experiment_config
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.config.resolve_store_root
    options:
      heading_level: 3
      show_source: false

## Root Contract

::: batgrad.ml.config.ExperimentConfig
    options:
      heading_level: 3
      members: false
      show_source: false

## Data And Loading

::: batgrad.ml.config.DataConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.ScalingRuleConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.LoaderTrainConfig
    options:
      heading_level: 3
      members: false
      show_source: false

## Training And Validation

::: batgrad.ml.config.TrainConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.MaskedSuffixConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.ValidationConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.ValidationSplitConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.ValidationGroupConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.ValidationMaskedSuffixConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.RolloutExtensionConfig
    options:
      heading_level: 3
      members: false
      show_source: false

## Runtime And Outputs

::: batgrad.ml.config.OptimizerConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.SchedulerConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.RunConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.LoggingConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.WandbConfig
    options:
      heading_level: 3
      members: false
      show_source: false

::: batgrad.ml.config.CheckpointConfig
    options:
      heading_level: 3
      members: false
      show_source: false
