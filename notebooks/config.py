# ruff: noqa: ANN001, ANN202, I002, INP001, PLC0415, PLR1711

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

with app.setup:
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
