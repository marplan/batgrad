from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebooks._support.config_helpers import (
    build_config,
    load_config_file,
    save_config_file,
)


def test_config_helpers_expose_typed_experiment_and_resolved_json() -> None:
    raw = json.loads(Path("configs/ml_dry_run_cpu.json").read_text(encoding="utf-8"))

    built = build_config(raw)

    assert built.error is None
    assert built.experiment is not None
    assert json.loads(built.json) == built.raw


def test_config_helpers_load_and_protect_existing_files(tmp_path: Path) -> None:
    source = Path("configs/ml_dry_run_cpu.json")
    loaded = load_config_file(str(source))
    assert loaded.load_error is None
    assert loaded.schema_error is None
    assert loaded.raw is not None

    destination = tmp_path / "config.json"
    save_config_file(str(destination), "{}", overwrite=False)
    with pytest.raises(FileExistsError):
        save_config_file(str(destination), "{}", overwrite=False)
