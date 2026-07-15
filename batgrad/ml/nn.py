from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from mamba_ssm.modules.mamba3 import Mamba3 as Mamba3Type

    from batgrad.ml.config import ExperimentConfig

SCALAR_INPUT_RANK = 3
BINNED_INPUT_RANK = 4

type LayerKind = Literal["attention", "ffn", "mamba", "reduce"]
type ResidualKind = Literal["standard", "none"]


@dataclass(frozen=True, slots=True)
class MambaConfig:
    """Default Mamba-3 layer parameters.

    Attributes:
        d_state: State-space dimension.
        expand: Inner expansion factor.
        headdim: Per-head width.
        ngroups: Number of state-space parameter groups.
        is_mimo: Enable Mamba-3 MIMO mode.
        mimo_rank: MIMO projection rank.
        chunk_size: Kernel processing chunk size.

    Note:
        Mamba layers require Linux, CUDA, and the `ml` dependency group.
        Cross-window state carry currently requires SISO mode (`is_mimo=False`).
    """

    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    is_mimo: bool = False
    mimo_rank: int = 1
    chunk_size: int = 64

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.d_state,
                self.expand,
                self.headdim,
                self.ngroups,
                self.mimo_rank,
                self.chunk_size,
            )
        ):
            raise ValueError("model.mamba values must be positive")


@dataclass(frozen=True, slots=True)
class ResidualConfig:
    """Residual connection policy for a configured layer.

    Attributes:
        kind: Standard additive residual or no residual.
    """

    kind: ResidualKind = "standard"

    def __post_init__(self) -> None:
        if self.kind not in {"standard", "none"}:
            raise ValueError("layer.residual.kind must be 'standard' or 'none'")


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """One feature-reduction or temporal-mixing layer.

    Attributes:
        kind: Attention, feed-forward, Mamba, or feature reduction.
        residual: Residual override. Temporal layers default to enabled and
            reduction defaults to disabled.
        bias: Optional attention/FFN linear-bias override. None inherits the
            model-level setting.
        mode: Reduction implementation; currently only `"sum_pool"`.
        d_state: Optional per-layer Mamba state dimension.
        expand: Optional per-layer Mamba expansion factor.
        headdim: Optional per-layer Mamba head width.
        ngroups: Optional per-layer Mamba group count.
        is_mimo: Optional per-layer Mamba MIMO mode.
        mimo_rank: Optional per-layer Mamba projection rank.
        chunk_size: Optional per-layer Mamba kernel chunk size.

    Note:
        The main layer sequence must contain exactly one reduction layer at index
        zero. Head layers cannot reduce. This validated transition changes
        `(B, T, C_in, D)` to `(B, T, D)` before temporal layers.
    """

    kind: LayerKind
    residual: bool | ResidualConfig | None = None
    bias: bool | None = None
    mode: Literal["sum_pool"] | None = None
    d_state: int | None = None
    expand: int | None = None
    headdim: int | None = None
    ngroups: int | None = None
    is_mimo: bool | None = None
    mimo_rank: int | None = None
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"attention", "ffn", "mamba", "reduce"}:
            raise ValueError(f"Unsupported model layer kind: {self.kind!r}")
        if self.kind == "reduce" and self.mode != "sum_pool":
            raise ValueError("reduce layers currently require mode='sum_pool'")
        if self.kind != "reduce" and self.mode is not None:
            raise ValueError("layer.mode is only valid for reduce layers")
        if self.kind not in {"attention", "ffn"} and self.bias is not None:
            raise ValueError("layer.bias is only valid for attention and ffn layers")
        if self.kind != "mamba" and any(
            value is not None
            for value in (
                self.d_state,
                self.expand,
                self.headdim,
                self.ngroups,
                self.is_mimo,
                self.mimo_rank,
                self.chunk_size,
            )
        ):
            raise ValueError("Mamba layer fields are only valid for kind='mamba'")

    @property
    def uses_residual(self) -> bool:
        """Return the effective residual policy for this layer."""
        if isinstance(self.residual, ResidualConfig):
            return self.residual.kind == "standard"
        if self.residual is not None:
            return bool(self.residual)
        return self.kind != "reduce"

    def linear_bias(self, *, default: bool) -> bool:
        """Return the effective linear-bias setting for this layer."""
        return default if self.bias is None else self.bias

    def mamba_config(self, default: MambaConfig) -> MambaConfig:
        """Merge per-layer Mamba overrides with model defaults.

        Args:
            default: Model-level Mamba parameters.

        Returns:
            A complete configuration for this layer.
        """
        return MambaConfig(
            d_state=default.d_state if self.d_state is None else self.d_state,
            expand=default.expand if self.expand is None else self.expand,
            headdim=default.headdim if self.headdim is None else self.headdim,
            ngroups=default.ngroups if self.ngroups is None else self.ngroups,
            is_mimo=default.is_mimo if self.is_mimo is None else self.is_mimo,
            mimo_rank=default.mimo_rank if self.mimo_rank is None else self.mimo_rank,
            chunk_size=default.chunk_size if self.chunk_size is None else self.chunk_size,
        )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Categorical output-head configuration.

    Attributes:
        parameterization: Output parameter sharing mode. Only `"shared"` is
            currently supported.
    """

    parameterization: Literal["shared"] = "shared"

    def __post_init__(self) -> None:
        if self.parameterization != "shared":
            raise ValueError("model.output.parameterization currently only supports 'shared'")


@dataclass(frozen=True, slots=True)
class SequenceMixerConfig:
    """Feature encoding and sequence-mixer architecture.

    Attributes:
        d_model: Temporal token width.
        n_heads: Attention head count; must divide `d_model`.
        mlp_ratio: Feed-forward hidden-width multiplier.
        dropout: Attention and feed-forward dropout probability.
        bias: Default bias setting for Batgrad-owned linear projections.
        norm: Normalization implementation; currently RMSNorm only.
        causal_attention: Prevent attention to future sequence positions.
        num_bins: Categorical bins used to encode each scalar.
        input_sigma: Gaussian width for input encoding; zero linearly interpolates
            between adjacent bins.
        output_sigma: Gaussian width for target distributions.
        feedback_mode: Reuse detached probabilities directly or decode and
            re-encode scalar feedback.
        mamba: Default Mamba parameters.
        layers: Main feature-reduction and temporal path.
        head_layers: Temporal layers after feature reduction.
        output: Categorical output projection settings.

    Examples:
        Build a small causal attention model:

        ```python
        from batgrad.ml.nn import LayerConfig, SequenceMixerConfig

        config = SequenceMixerConfig(
            d_model=128,
            n_heads=4,
            layers=(
                LayerConfig(kind="reduce", mode="sum_pool"),
                LayerConfig(kind="attention"),
                LayerConfig(kind="ffn"),
            ),
        )
        ```
    """

    d_model: int = 256
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bias: bool = False
    norm: Literal["rmsnorm"] = "rmsnorm"
    causal_attention: bool = True
    num_bins: int = 64
    input_sigma: float = 0.0
    output_sigma: float = 0.0
    feedback_mode: Literal["probabilities", "decoded_scalar"] = "probabilities"
    mamba: MambaConfig = field(default_factory=MambaConfig)
    layers: tuple[LayerConfig, ...] = (
        LayerConfig(kind="reduce", mode="sum_pool"),
        LayerConfig(kind="attention"),
        LayerConfig(kind="ffn"),
    )
    head_layers: tuple[LayerConfig, ...] = (LayerConfig(kind="ffn"),)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("model.d_model must be positive and divisible by model.n_heads")
        if self.mlp_ratio <= 0.0:
            raise ValueError("model.mlp_ratio must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("model.dropout must be in [0, 1)")
        if self.norm != "rmsnorm":
            raise ValueError("model.norm currently only supports 'rmsnorm'")
        if self.num_bins <= 1:
            raise ValueError("model.num_bins must be > 1")
        if self.input_sigma < 0.0 or self.output_sigma < 0.0:
            raise ValueError("model input_sigma and output_sigma must be >= 0")
        if self.feedback_mode not in {"probabilities", "decoded_scalar"}:
            raise ValueError("model.feedback_mode must be 'probabilities' or 'decoded_scalar'")
        if not self.layers:
            raise ValueError("model.layers must not be empty")
        _validate_layer_topology(self)


def _validate_layer_topology(config: SequenceMixerConfig) -> None:
    reduce_indices = tuple(
        index for index, layer in enumerate(config.layers) if layer.kind == "reduce"
    )
    if reduce_indices != (0,):
        raise ValueError("model.layers must contain exactly one reduce layer at index 0")
    if any(layer.kind == "reduce" for layer in config.head_layers):
        raise ValueError("model.head_layers must not contain reduce layers")


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float, *, bias: bool) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, hidden, bias=bias),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, config: SequenceMixerConfig, layer: LayerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.causal = config.causal_attention
        self.norm = nn.RMSNorm(config.d_model)
        bias = layer.linear_bias(default=config.bias)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=bias)
        self.out = nn.Linear(config.d_model, config.d_model, bias=bias)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(self.norm(x)).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        attn_mask = None
        use_causal = self.causal
        if mask is not None:
            valid = mask.to(dtype=torch.bool, device=x.device)
            query_valid = valid[:, None, :, None]
            key_valid = valid[:, None, None, :]
            attn_mask = torch.logical_and(query_valid, key_valid).expand(
                batch, self.n_heads, seq_len, seq_len
            )
            if self.causal:
                causal_mask = torch.ones(
                    (seq_len, seq_len), dtype=torch.bool, device=x.device
                ).tril()
                attn_mask = torch.logical_and(attn_mask, causal_mask.view(1, 1, seq_len, seq_len))
                use_causal = False
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )
        return self.out(y.transpose(1, 2).reshape(batch, seq_len, width))


@dataclass(frozen=True, slots=True)
class MambaCarryState:
    """Opaque recurrent state for one SISO Mamba-3 layer.

    State is lane- and window-position-specific; callers must never reuse it
    across unrelated streams or misaligned window starts.
    """

    angle_state: torch.Tensor
    ssm_state: torch.Tensor
    k_state: torch.Tensor
    v_state: torch.Tensor

    def detach(self) -> MambaCarryState:
        return MambaCarryState(
            angle_state=self.angle_state.detach(),
            ssm_state=self.ssm_state.detach(),
            k_state=self.k_state.detach(),
            v_state=self.v_state.detach(),
        )


class MambaBlock(nn.Module):
    mamba: Mamba3Type

    def __init__(
        self, config: SequenceMixerConfig, layer_config: LayerConfig, device: torch.device
    ) -> None:
        super().__init__()
        if device.type != "cuda":
            raise ValueError(
                "Mamba layers are currently supported only on CUDA for batgrad ML runs"
            )
        from mamba_ssm.modules.mamba3 import Mamba3  # noqa: PLC0415

        mamba_config = layer_config.mamba_config(config.mamba)
        self.norm = nn.RMSNorm(config.d_model)
        self.mamba = Mamba3(
            d_model=config.d_model,
            d_state=mamba_config.d_state,
            expand=mamba_config.expand,
            headdim=mamba_config.headdim,
            ngroups=mamba_config.ngroups,
            is_mimo=mamba_config.is_mimo,
            mimo_rank=mamba_config.mimo_rank,
            chunk_size=mamba_config.chunk_size,
        )
        self.is_mimo = mamba_config.is_mimo

    @property
    def supports_stateful_windows(self) -> bool:
        return not self.is_mimo

    def _forward_siso(
        self,
        x: torch.Tensor,
        state: MambaCarryState | None,
        *,
        return_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, MambaCarryState]:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (  # noqa: PLC0415
            mamba3_siso_combined,
        )

        mixer = self.mamba
        zxbcdt_attrap = mixer.in_proj(x)
        z, x_proj, b_state, c_state, dd_dt, dd_a, trap, angles = torch.split(
            zxbcdt_attrap,
            [
                mixer.d_inner,
                mixer.d_inner,
                mixer.d_state * mixer.num_bc_heads * mixer.mimo_rank,
                mixer.d_state * mixer.num_bc_heads * mixer.mimo_rank,
                mixer.nheads,
                mixer.nheads,
                mixer.nheads,
                mixer.num_rope_angles,
            ],
            dim=-1,
        )
        batch, seq_len = x.shape[:2]
        z = z.view(batch, seq_len, mixer.nheads, mixer.headdim)
        x_proj = x_proj.view(batch, seq_len, mixer.nheads, mixer.headdim)
        b_state = b_state.view(batch, seq_len, mixer.mimo_rank, mixer.num_bc_heads, mixer.d_state)
        c_state = c_state.view(batch, seq_len, mixer.mimo_rank, mixer.num_bc_heads, mixer.d_state)
        trap = trap.transpose(1, 2)

        a_decay = -functional.softplus(dd_a.to(torch.float32)).clamp(min=mixer.A_floor)
        dt = functional.softplus(dd_dt + mixer.dt_bias)
        adt = a_decay * dt
        dt = dt.transpose(1, 2)
        adt = adt.transpose(1, 2)

        angles = angles.unsqueeze(-2).expand(-1, -1, mixer.nheads, -1).to(torch.float32)
        b_state = mixer.B_norm(b_state)
        c_state = mixer.C_norm(c_state)
        input_states = None
        if state is not None:
            input_states = (
                state.angle_state,
                state.ssm_state,
                state.k_state.squeeze(1),
                state.v_state,
            )

        y_result = mamba3_siso_combined(
            Q=c_state.squeeze(2),
            K=b_state.squeeze(2),
            V=x_proj,
            ADT=adt,
            DT=dt,
            Trap=trap,
            Q_bias=mixer.C_bias.squeeze(1),
            K_bias=mixer.B_bias.squeeze(1),
            Angles=angles,
            D=mixer.D,
            Z=z if not mixer.is_outproj_norm else None,
            Input_States=input_states,
            chunk_size=mixer.chunk_size,
            return_final_states=return_state,
            cu_seqlens=None,
        )

        next_state: MambaCarryState | None = None
        if return_state:
            y, last_angle, last_state, last_k, last_v = cast(
                "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]",
                y_result,
            )
        else:
            y = cast("torch.Tensor", y_result)
        if return_state:
            next_state = MambaCarryState(
                angle_state=last_angle,
                ssm_state=last_state,
                k_state=last_k.unsqueeze(1),
                v_state=last_v,
            )
        y = y.reshape(batch, seq_len, mixer.nheads * mixer.headdim)
        if mixer.is_outproj_norm:
            z = z.reshape(batch, seq_len, mixer.nheads * mixer.headdim)
            y = mixer.norm(y, z)
        out = mixer.out_proj(y.to(x_proj.dtype))
        if next_state is not None:
            return out, next_state
        return out

    @torch.compiler.disable(recursive=True)
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        state: MambaCarryState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MambaCarryState]:
        del mask
        x = self.norm(x)
        if state is not None or return_state:
            if self.is_mimo:
                raise ValueError("stateful Mamba windows currently support SISO only")
            return self._forward_siso(x, state, return_state=return_state)
        return self.mamba(x)


class SequenceMixer(nn.Module):
    """Categorical multi-feature sequence model.

    The module consumes pre-encoded inputs shaped `(B, T, C_in, K)` and emits
    logits shaped `(B, T, C_out, K)`. Use `encode_categorical_values` for
    scalar inputs or the experiment helpers used by training and rollout.

    Args:
        config: Model architecture.
        input_dim: Number of ordered input features.
        output_dim: Number of ordered targets.
        device: Construction device. Mamba layers require CUDA.
    """

    def __init__(
        self,
        config: SequenceMixerConfig,
        input_dim: int,
        output_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_bins = config.num_bins
        self.feature_proj = nn.ModuleList(
            [nn.Linear(config.num_bins, config.d_model, bias=config.bias) for _ in range(input_dim)]
        )
        self.layers = nn.ModuleList([self._build_layer(layer, device) for layer in config.layers])
        self.head_layers = nn.ModuleList(
            [self._build_layer(layer, device) for layer in config.head_layers]
        )
        self.final_norm = nn.RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, output_dim * config.num_bins, bias=config.bias)

    def _build_layer(self, layer: LayerConfig, device: torch.device) -> nn.Module:
        if layer.kind == "attention":
            return SelfAttention(self.config, layer)
        if layer.kind == "ffn":
            return FeedForward(
                self.config.d_model,
                self.config.mlp_ratio,
                self.config.dropout,
                bias=layer.linear_bias(default=self.config.bias),
            )
        if layer.kind == "mamba":
            return MambaBlock(self.config, layer, device)
        return nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        states: dict[str, MambaCarryState] | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, MambaCarryState]]:
        """Calculate categorical logits and optional recurrent states.

        Args:
            x: Encoded inputs shaped `(B, T, C_in, K)`.
            mask: Optional valid-row mask shaped `(B, T)`.
            states: Recurrent states keyed by configured Mamba layer path.
            return_states: Return final state for each stateful layer.

        Returns:
            Logits shaped `(B, T, C_out, K)`, optionally paired with recurrent
            states aligned to the end of the supplied sequence.
        """
        if x.ndim != BINNED_INPUT_RANK:
            raise ValueError(
                f"SequenceMixer expects binned inputs shaped (B,T,C,K), got {tuple(x.shape)}"
            )
        if int(x.shape[-2]) != self.input_dim:
            raise ValueError(
                f"input channel mismatch: expected {self.input_dim}, got {x.shape[-2]}"
            )
        if int(x.shape[-1]) != self.num_bins:
            raise ValueError(f"input bin mismatch: expected {self.num_bins}, got {x.shape[-1]}")
        h = torch.stack(
            [proj(x[:, :, idx, :]) for idx, proj in enumerate(self.feature_proj)],
            dim=2,
        )
        next_states: dict[str, MambaCarryState] = {}
        for layer_idx, (spec, layer) in enumerate(
            zip(self.config.layers, self.layers, strict=True)
        ):
            if spec.kind == "reduce":
                h = h.sum(dim=2)
                continue
            layer_key = f"layers.{layer_idx}"
            if spec.kind == "mamba" and return_states:
                output, state = cast("MambaBlock", layer)(
                    h,
                    mask,
                    state=None if states is None else states.get(layer_key),
                    return_state=True,
                )
                next_states[layer_key] = state
                y = output
            else:
                y = layer(h, mask)
            h = h + y if spec.uses_residual else y
        for layer_idx, (spec, layer) in enumerate(
            zip(self.config.head_layers, self.head_layers, strict=True)
        ):
            layer_key = f"head_layers.{layer_idx}"
            if spec.kind == "mamba" and return_states:
                output, state = cast("MambaBlock", layer)(
                    h,
                    mask,
                    state=None if states is None else states.get(layer_key),
                    return_state=True,
                )
                next_states[layer_key] = state
                y = output
            else:
                y = layer(h, mask)
            h = h + y if spec.uses_residual else y
        logits = self.output(self.final_norm(h)).view(*h.shape[:2], self.output_dim, self.num_bins)
        if return_states:
            return logits, next_states
        return logits


def build_model(config: ExperimentConfig, device: torch.device) -> SequenceMixer:
    """Build an experiment's model on the requested device.

    Args:
        config: Validated experiment configuration defining architecture and
            ordered input/target columns.
        device: Destination device.

    Returns:
        An initialized sequence mixer on `device`.

    Raises:
        ValueError: If a Mamba model is requested on a non-CUDA device.
    """
    return SequenceMixer(
        config.model,
        input_dim=len(config.data.input_columns),
        output_dim=len(config.data.target_columns),
        device=device,
    ).to(device)


def categorical_target_distribution(
    target: torch.Tensor,
    num_bins: int,
    sigma: float,
    target_ranges: torch.Tensor,
) -> torch.Tensor:
    """Encode scalar targets as distributions over bounded categorical bins.

    With `sigma=0`, probability mass is linearly interpolated between adjacent
    bins. Positive sigma produces a normalized Gaussian over bin centers. Values
    outside their target ranges are clamped before encoding.

    Args:
        target: Values shaped `(..., C)`.
        num_bins: Number of categorical bins.
        sigma: Gaussian width in normalized `[0, 1]` coordinates.
        target_ranges: Per-channel bounds shaped `(C, 2)`.

    Returns:
        Float32 distributions shaped `(..., C, K)`.
    """
    if sigma < 0.0:
        raise ValueError("categorical sigma must be >= 0")
    if target_ranges.shape != (int(target.shape[-1]), 2):
        raise ValueError(
            "target_ranges must have shape (C, 2), "
            f"got {tuple(target_ranges.shape)} for target shape {tuple(target.shape)}"
        )
    ranges = target_ranges.to(device=target.device, dtype=torch.float32)
    value_min = ranges[:, 0].view(*((1,) * (target.ndim - 1)), -1)
    value_max = ranges[:, 1].view(*((1,) * (target.ndim - 1)), -1)
    values = target.float().clamp(value_min, value_max)
    normalized = (values - value_min) / (value_max - value_min).clamp_min(
        torch.finfo(torch.float32).eps
    )
    if sigma == 0.0:
        position = normalized * float(num_bins - 1)
        lower = torch.floor(position).to(torch.long)
        upper = torch.ceil(position).to(torch.long)
        upper_weight = position - lower.to(position.dtype)
        result = torch.zeros((*target.shape, num_bins), dtype=torch.float32, device=target.device)
        result.scatter_add_(-1, lower.unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
        result.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return result
    bins = torch.linspace(0.0, 1.0, num_bins, dtype=torch.float32, device=target.device)
    dist = torch.exp(-torch.square(normalized.unsqueeze(-1) - bins) / (2.0 * sigma * sigma))
    return dist / dist.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)


def encode_categorical_values(
    values: torch.Tensor,
    num_bins: int,
    sigma: float,
    value_ranges: torch.Tensor,
) -> torch.Tensor:
    """Encode scalar model inputs as categorical distributions.

    Args:
        values: Scalar inputs shaped `(B, T, C)`.
        num_bins: Number of categorical bins.
        sigma: Gaussian width in normalized coordinates; zero uses adjacent-bin
            interpolation.
        value_ranges: Per-channel bounds shaped `(C, 2)`.

    Returns:
        Encoded values shaped `(B, T, C, K)` with the input dtype.

    Raises:
        ValueError: If input rank or range shape is invalid.
    """
    if values.ndim != SCALAR_INPUT_RANK:
        raise ValueError(f"categorical inputs must be shaped (B,T,C), got {tuple(values.shape)}")
    return categorical_target_distribution(values, num_bins, sigma, value_ranges).to(
        dtype=values.dtype
    )


def decode_categorical_logits(logits: torch.Tensor, target_ranges: torch.Tensor) -> torch.Tensor:
    """Decode logits through the expected bin location.

    This returns the probability-weighted expectation, not the argmax bin.

    Args:
        logits: Categorical logits shaped `(..., C, K)`.
        target_ranges: Per-channel output bounds shaped `(C, 2)`.

    Returns:
        Decoded values shaped `(..., C)` in output-scaled units.
    """
    ranges = target_ranges.to(device=logits.device, dtype=torch.float32)
    if ranges.shape != (int(logits.shape[-2]), 2):
        raise ValueError(
            "target_ranges must have shape (C, 2), "
            f"got {tuple(ranges.shape)} for logits shape {tuple(logits.shape)}"
        )
    normalized_bins = torch.linspace(
        0.0, 1.0, int(logits.shape[-1]), dtype=torch.float32, device=logits.device
    )
    decoded = (torch.softmax(logits.float(), dim=-1) * normalized_bins).sum(dim=-1)
    value_min = ranges[:, 0].view(*((1,) * (decoded.ndim - 1)), -1)
    value_max = ranges[:, 1].view(*((1,) * (decoded.ndim - 1)), -1)
    return (value_min + decoded * (value_max - value_min)).to(dtype=logits.dtype)
