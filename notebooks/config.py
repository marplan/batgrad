# ruff: noqa: ANN001, ANN202, I002, INP001, PLC0415, PLR1711, S603, S607

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
    import os
    import subprocess
    import sys
    from pathlib import Path

    def is_batgrad_root(root: Path) -> bool:
        return (
            (root / "pyproject.toml").is_file()
            and (root / "batgrad" / "__init__.py").is_file()
            and (root / "notebooks" / "_support" / "config_editor.py").is_file()
        )

    local_root = Path(__file__).resolve().parents[1]
    if not is_batgrad_root(local_root):
        local_root = Path("/marimo/batgrad")
        if not is_batgrad_root(local_root):
            local_root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/marplan/batgrad.git",
                    str(local_root),
                ],
                check=True,
            )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--editable",
                str(local_root),
            ],
            check=True,
            cwd=local_root,
        )

    project_root = local_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    os.environ.setdefault("DATA_ROOT", "/marimo/data")

    import marimo as mo


@app.cell
def _():
    from notebooks._support.config_editor import app as config_editor_app

    return (config_editor_app,)


@app.cell
async def _(config_editor_app):
    config_editor_result = await config_editor_app.embed()
    config_editor = config_editor_app
    return config_editor, config_editor_result


@app.cell
def _(config_editor_result):
    loaded_config = config_editor_result.defs.get("loaded_config")
    generated_config = config_editor_result.defs.get("generated_config", {})
    experiment_config = config_editor_result.defs.get("experiment_config")
    generated_json = config_editor_result.defs.get("generated_json", "{}")
    validation_error = config_editor_result.defs.get("validation_error")
    return (
        experiment_config,
        generated_config,
        generated_json,
        loaded_config,
        validation_error,
    )


@app.cell
def _(
    config_editor,
    config_editor_result,
    experiment_config,
    generated_config,
    generated_json,
    loaded_config,
    validation_error,
):
    _ = (
        config_editor,
        experiment_config,
        generated_config,
        generated_json,
        loaded_config,
        validation_error,
    )
    mo.vstack(
        [
            config_editor_result.output,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
