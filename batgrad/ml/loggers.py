from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from batgrad.logging import get_logger

logger = get_logger(__name__)

LogValue = int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class WandbConfig:
    project: str | None = None
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    backend: Literal["stdout", "jsonl", "wandb"] = "jsonl"
    mode: Literal["offline", "online"] = "offline"
    mirror_stdout: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        if self.backend not in {"stdout", "jsonl", "wandb"}:
            raise ValueError(f"Unsupported logging backend: {self.backend!r}")
        if self.mode not in {"offline", "online"}:
            raise ValueError(f"Unsupported logging mode: {self.mode!r}")


class RunLogger(Protocol):
    def log_metrics(
        self,
        step: int,
        metrics: dict[str, LogValue],
        *,
        epoch: int | None = None,
        epoch_pct: int | None = None,
    ) -> None: ...
    def log_payload(self, step: int, name: str, payload: object) -> None: ...
    def run_name(self) -> str | None: ...
    def finish(self) -> None: ...


class StdoutRunLogger:
    def log_metrics(
        self,
        step: int,
        metrics: dict[str, LogValue],
        *,
        epoch: int | None = None,
        epoch_pct: int | None = None,
    ) -> None:
        rendered = " ".join(
            f"{key}={_format_metric_value(key, value)}" for key, value in metrics.items()
        )
        logger.info("step=%03d%s %s", step, _format_progress(epoch, epoch_pct), rendered)

    def log_payload(self, step: int, name: str, payload: object) -> None:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        logger.info("step=%03d payload=%s keys=%s", step, name, keys)

    def run_name(self) -> str | None:
        return None

    def finish(self) -> None:
        return


class JsonlRunLogger:
    def __init__(self, run_dir: Path) -> None:
        self._run_name = run_dir.name
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (log_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self.payload_file = (log_dir / "payloads.jsonl").open("a", encoding="utf-8")

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, LogValue],
        *,
        epoch: int | None = None,
        epoch_pct: int | None = None,
    ) -> None:
        progress = _progress_payload(epoch, epoch_pct)
        self.metrics_file.write(json.dumps({"step": step, **progress, **metrics}) + "\n")
        self.metrics_file.flush()

    def log_payload(self, step: int, name: str, payload: object) -> None:
        self.payload_file.write(
            json.dumps({"step": step, "name": name, "payload": _json_payload(payload)}) + "\n"
        )
        self.payload_file.flush()

    def run_name(self) -> str | None:
        return self._run_name

    def finish(self) -> None:
        self.metrics_file.close()
        self.payload_file.close()


class WandbRunLogger:
    def __init__(self, config: LoggingConfig, run_dir: Path, run_config: dict[str, object]) -> None:
        import wandb  # noqa: PLC0415 - W&B is only required for the W&B logging backend.

        os.environ.setdefault("WANDB_DIR", str(run_dir / "wandb"))
        Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            tags=list(config.wandb.tags),
            mode=config.mode,
            dir=str(run_dir),
            config=run_config,
        )

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, LogValue],
        *,
        epoch: int | None = None,
        epoch_pct: int | None = None,
    ) -> None:
        self.run.log({**_progress_payload(epoch, epoch_pct), **metrics}, step=step)

    def log_payload(self, step: int, name: str, payload: object) -> None:
        self.run.log({name: payload}, step=step)

    def run_name(self) -> str | None:
        name = getattr(self.run, "name", None)
        return str(name) if name else None

    def finish(self) -> None:
        self.run.finish()


class CompositeRunLogger:
    def __init__(self, loggers: tuple[RunLogger, ...]) -> None:
        self.loggers = loggers

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, LogValue],
        *,
        epoch: int | None = None,
        epoch_pct: int | None = None,
    ) -> None:
        for logger in self.loggers:
            logger.log_metrics(step, metrics, epoch=epoch, epoch_pct=epoch_pct)

    def log_payload(self, step: int, name: str, payload: object) -> None:
        for logger in self.loggers:
            logger.log_payload(step, name, payload)

    def run_name(self) -> str | None:
        for logger in reversed(self.loggers):
            name = logger.run_name()
            if name:
                return name
        return None

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()


def build_logger(
    config: LoggingConfig, run_dir: Path | None, run_config: dict[str, object]
) -> RunLogger:
    if config.backend == "stdout":
        return StdoutRunLogger()
    if run_dir is None:
        raise ValueError("file-backed logging requires run.output_dir")
    base: RunLogger = (
        JsonlRunLogger(run_dir)
        if config.backend == "jsonl"
        else WandbRunLogger(config, run_dir, run_config)
    )
    if config.mirror_stdout and config.backend != "stdout":
        return CompositeRunLogger((StdoutRunLogger(), base))
    return base


def _format_progress(epoch: int | None, epoch_pct: int | None) -> str:
    if epoch is None or epoch_pct is None:
        return ""
    return f" epoch={epoch}:{epoch_pct:3d}%"


def _progress_payload(epoch: int | None, epoch_pct: int | None) -> dict[str, int]:
    payload: dict[str, int] = {}
    if epoch is not None:
        payload["epoch"] = epoch
    if epoch_pct is not None:
        payload["epoch_pct"] = epoch_pct
    return payload


def _json_payload(payload: object) -> object:
    to_plotly_json = getattr(payload, "to_plotly_json", None)
    if callable(to_plotly_json):
        from plotly.utils import PlotlyJSONEncoder  # noqa: PLC0415

        return json.loads(json.dumps(to_plotly_json(), cls=PlotlyJSONEncoder))
    return payload


def _format_metric_value(key: str, value: LogValue) -> str:
    if isinstance(value, bool | str) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:03d}"
    key_lower = key.lower()
    if "lr" in key_lower:
        return f"{value:.3e}"
    if "loss" in key_lower:
        return f"{value:.3f}"
    return f"{value:.3g}"
