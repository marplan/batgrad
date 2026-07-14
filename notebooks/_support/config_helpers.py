from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from batgrad.ml.config import ExperimentConfig, config_to_dict, parse_experiment_config


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    raw: dict[str, object] | None
    load_error: str | None
    schema_error: str | None


@dataclass(frozen=True, slots=True)
class BuiltConfig:
    raw: dict[str, object]
    experiment: ExperimentConfig | None
    json: str
    error: str | None


def scaling_drafts(
    columns: tuple[str, ...],
    configured: tuple[dict[str, object], ...] = (),
) -> tuple[dict[str, object], ...]:
    """Build editable explicit scaling drafts from loaded rules."""
    configured_by_column = {
        str(item.get("column")): item
        for item in configured
        if str(item.get("column", "")).strip()
    }
    drafts: list[dict[str, object]] = []
    for column in dict.fromkeys(column.strip() for column in columns if column.strip()):
        draft: dict[str, object] = {
            "column": column,
            "input_min": "",
            "input_max": "",
            "output_min": -1.0,
            "output_max": 1.0,
            "clip": False,
            "transform": "linear",
        }
        draft.update(configured_by_column.get(column, {}))
        draft["column"] = column
        drafts.append(draft)
    return tuple(drafts)


def load_config_file(path_value: str) -> LoadedConfig:
    path = Path(path_value).expanduser()
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            return LoadedConfig(None, "Config root must be a JSON object", None)
        try:
            parse_experiment_config(raw)
            schema_error = None
        except (TypeError, ValueError) as exc:
            schema_error = str(exc)
        return LoadedConfig(raw, None, schema_error)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return LoadedConfig(None, str(exc), None)


def build_config(raw: dict[str, object], errors: list[str] | tuple[str, ...] = ()) -> BuiltConfig:
    if errors:
        message = "; ".join(errors)
        return BuiltConfig({}, None, "{}", message)
    try:
        experiment = parse_experiment_config(raw)
    except (TypeError, ValueError) as exc:
        return BuiltConfig(raw, None, json.dumps(raw, indent=2), str(exc))
    resolved = config_to_dict(experiment)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved experiment config must be an object")
    return BuiltConfig(resolved, experiment, json.dumps(resolved, indent=2), None)


def save_config_file(path_value: str, generated_json: str, *, overwrite: bool) -> Path:
    path = Path(path_value).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_json + "\n", encoding="utf-8")
    return path
