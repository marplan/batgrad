# ML Models

The built-in sequence mixer represents scalar values as categorical
distributions, projects each input feature independently, reduces the feature
axis, and applies configured temporal layers. The main path must contain exactly
one reduction layer at index zero, and the head cannot contain reduction layers.
Mamba layers require Linux, CUDA, and the `ml` dependency group.

::: batgrad.ml.nn.SequenceMixerConfig
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.nn.LayerConfig
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.nn.ResidualConfig
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.nn.MambaConfig
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.nn.OutputConfig
    options:
      heading_level: 2
      members: false
      show_source: false

::: batgrad.ml.nn.build_model
    options:
      heading_level: 2
      show_source: false

::: batgrad.ml.nn.SequenceMixer
    options:
      heading_level: 2
      members:
        - forward
      show_source: false

## Advanced Recurrent State

`MambaCarryState` is an opaque advanced type exposed by stateful model execution.
Application code should pass it back only to the same model layer and aligned
stream; it should not inspect or share its tensors across unrelated samples.

::: batgrad.ml.nn.MambaCarryState
    options:
      heading_level: 3
      members: false
      show_source: false

## Categorical Encoding

::: batgrad.ml.nn.categorical_target_distribution
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.nn.encode_categorical_values
    options:
      heading_level: 3
      show_source: false

::: batgrad.ml.nn.decode_categorical_logits
    options:
      heading_level: 3
      show_source: false
