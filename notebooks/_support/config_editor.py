# ruff: noqa: ANN001, ANN202, C901, E501, FBT003, I002, PLR0911, PLR0915, PLR1711, PLW2901, Q001, SIM212, TRY301

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import json
    import math
    from pathlib import Path

    import marimo as mo

    from notebooks._support.config_helpers import (
        build_config,
        load_config_file,
        save_config_file,
        scaling_drafts,
    )


@app.cell
def _():
    config_paths = tuple(str(path) for path in sorted(Path("configs").rglob("*.json")))
    default_load_path = "configs/ml_baseline.json"
    load_options = config_paths or (default_load_path,)
    load_path = mo.ui.dropdown(
        options=load_options,
        value=default_load_path if default_load_path in load_options else load_options[0],
        label="Load config",
    )
    save_path = mo.ui.text(
        value="configs/ml_generated.json",
        label="Save config path",
    )
    overwrite_existing = mo.ui.checkbox(
        value=False,
        label="Allow overwrite existing file",
    )
    return load_path, overwrite_existing, save_path


@app.cell
def _():
    get_action_status, set_action_status = mo.state(None)
    get_form_drafts, set_form_drafts = mo.state({})
    get_layer_kind_change, set_layer_kind_change = mo.state(None)
    return (
        get_action_status,
        get_form_drafts,
        get_layer_kind_change,
        set_action_status,
        set_form_drafts,
        set_layer_kind_change,
    )


@app.cell
def _(load_path):
    _loaded = load_config_file(str(load_path.value))
    loaded_config = _loaded.raw
    load_error = _loaded.load_error
    loaded_schema_error = _loaded.schema_error
    return load_error, loaded_config, loaded_schema_error


@app.cell
def _():
    def path_value(config: dict[str, object] | None, path: str, default):
        current = config or {}
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def json_value(value: object) -> str:
        return json.dumps(value, indent=2, sort_keys=False)

    def lines_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, list | tuple):
            return "\n".join(str(item) for item in value)
        return str(value)

    def parse_lines(value: str) -> list[str]:
        return [line.strip() for line in value.splitlines() if line.strip()]

    def parse_json_field(path: str, value: str):
        try:
            return json.loads(value or "null"), None
        except json.JSONDecodeError as exc:
            return None, f"{path}: {exc}"

    def parse_nullable_int(value: object):
        text = str(value).strip()
        return None if not text else int(text)

    def parse_nullable_str(value: object):
        text = str(value).strip()
        return None if not text else text

    def nonempty_values(values: object) -> list[str]:
        if not isinstance(values, list | tuple):
            return []
        result = []
        for item in values:
            value = item.value if hasattr(item, "value") else item
            text = str(value).strip()
            if text:
                result.append(text)
        return result

    def parse_json_value(path: str, value: object):
        text = str(value).strip()
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            return None, f"{path}: expected a JSON value: {exc.msg}"

    def dict_entry_values(path: str, entries: object):
        if not isinstance(entries, list | tuple):
            return {}, []
        result = {}
        errors = []
        for index, (key_widget, value_widget) in enumerate(entries):
            key = str(key_widget.value).strip()
            if key:
                value, error = parse_json_value(
                    f"{path}[{index}].match[{key!r}]", value_widget.value
                )
                if error is not None:
                    errors.append(error)
                else:
                    result[key] = value
        return result, errors

    def float_dict_entry_values(path: str, entries: object) -> dict[str, float]:
        if not isinstance(entries, list | tuple):
            return {}
        result = {}
        for index, (key_widget, value_widget) in enumerate(entries):
            key = str(key_widget.value).strip()
            if key:
                try:
                    result[key] = float(value_widget.value)
                except ValueError as exc:
                    raise ValueError(f"{path}[{index}] {key!r}: expected a float") from exc
        return result

    def manifest_values(values: object):
        if not isinstance(values, list | tuple):
            return {}, ["data.manifest_paths must contain JSON object entries"]
        result = {}
        errors = []
        for index, item in enumerate(values):
            if hasattr(item, "value"):
                item = item.value
            text = str(item).strip().rstrip(",")
            if not text:
                errors.append(f"data.manifest_paths[{index}]: row must not be blank")
                continue
            try:
                parsed = json.loads("{" + text + "}")
            except json.JSONDecodeError as exc:
                errors.append(
                    f"data.manifest_paths[{index}]: malformed JSON object entry: {exc.msg}"
                )
                continue
            if len(parsed) != 1:
                errors.append(f"data.manifest_paths[{index}]: row must contain exactly one path")
                continue
            path, commit = next(iter(parsed.items()))
            if not isinstance(commit, str) or not path.strip() or not commit.strip():
                errors.append(
                    f"data.manifest_paths[{index}]: path and commit must be nonblank strings"
                )
                continue
            if path in result:
                errors.append(f"data.manifest_paths[{index}]: duplicate path {path!r}")
                continue
            result[path] = commit
        return result, errors

    def scaling_rule_values(entries: object):
        if not isinstance(entries, list | tuple):
            return [], ["data.scaling must contain one rule per selected column"]
        result = []
        errors = []
        for index, entry in enumerate(entries):
            column = str(entry["column"])
            try:
                input_min = float(entry["input_min"].value)
                input_max = float(entry["input_max"].value)
                output_min = float(entry["output_min"].value)
                output_max = float(entry["output_max"].value)
            except (TypeError, ValueError):
                errors.append(
                    f"data.scaling[{index}] {column!r}: bounds must be finite numbers"
                )
                continue
            if not all(math.isfinite(value) for value in (input_min, input_max, output_min, output_max)):
                errors.append(
                    f"data.scaling[{index}] {column!r}: bounds must be finite numbers"
                )
                continue
            result.append(
                {
                    "column": column,
                    "input_min": input_min,
                    "input_max": input_max,
                    "output_min": output_min,
                    "output_max": output_max,
                    "clip": bool(entry["clip"].value),
                    "transform": str(entry["transform"].value),
                }
            )
        return result, errors

    def nullable_bool_value(value: object) -> str:
        if isinstance(value, str) and value in {"inherit/null", "true", "false"}:
            return str(value)
        if value is None:
            return "inherit/null"
        return "true" if bool(value) else "false"

    def parse_nullable_bool(value: object):
        if value == "inherit/null":
            return None
        return value == "true"

    def set_path(config: dict[str, object], path: str, value: object) -> None:
        current = config
        parts = path.split(".")
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current[parts[-1]] = value

    return (
        dict_entry_values,
        float_dict_entry_values,
        manifest_values,
        nonempty_values,
        nullable_bool_value,
        parse_nullable_bool,
        parse_nullable_int,
        parse_nullable_str,
        path_value,
        scaling_rule_values,
        set_path,
    )


@app.cell
def _():
    def safe_html_attr(value: str) -> str:
        return (
            value.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def info_icon(tooltip: str):
        safe_tooltip = safe_html_attr(tooltip)
        return mo.Html(
            f'''<span
              data-tooltip="{safe_tooltip}"
              style="
                color: rgb(70, 167, 88);
                background-color: transparent;
                border: 1px solid rgb(70, 167, 88);
                width: 16px;
                height: 16px;
                box-sizing: border-box;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                line-height: 1;
                font-weight: bold;
                cursor: help;
                margin-inline: 0.375rem;
                flex: 0 0 auto;
                text-decoration: none;
              "
            >i</span>'''
        )

    def with_help(widget: object, tooltip: str | None = None, *, align: str = "center"):
        if tooltip is None:
            return widget
        return mo.hstack(
            [widget, info_icon(tooltip)],
            justify="start",
            gap=0.5,
            align=align,
        )

    def dynamic_array(array_widget: object, count_widget: object, tooltip: str | None = None):
        if isinstance(array_widget, list | tuple):
            array_widget = mo.vstack(list(array_widget), gap=0.15)
        items = [array_widget]
        if tooltip is not None:
            items.append(info_icon(tooltip))
        items.append(count_widget.style({"width": "3rem"}))
        return mo.hstack(items, justify="start", gap=0.5, align="start")

    def tree_array_rows(
        row_widgets: tuple[object, ...], count_widget: object, tooltip: str | None = None
    ):
        rows = list(row_widgets)
        first_row_items = [] if not rows else [rows[0]]
        if tooltip is not None:
            first_row_items.append(info_icon(tooltip))
        first_row_items.append(count_widget.style({"width": "3rem"}))
        first_row = mo.hstack(
            first_row_items,
            justify="start",
            gap=0.5,
            align="center",
        )
        return [first_row, *rows[1:]]

    def dict_entry_widgets(
        values: object,
        count: int,
        *,
        key_placeholder: str = "key",
        value_placeholder: str = "value",
        on_change=None,
    ):
        items = (
            list(values.items())
            if isinstance(values, dict)
            else list(values)
            if isinstance(values, list | tuple)
            else []
        )

        def item_value(index: int) -> str:
            if index >= len(items):
                return ""
            value = items[index][1]
            return json.dumps(value) if isinstance(values, dict) else str(value)

        def update_at(index: int, part: int):
            def update(value: str):
                if on_change is None:
                    return
                current = [
                    [str(key), item_value(item_index)] for item_index, (key, _) in enumerate(items)
                ]
                while len(current) < count:
                    current.append(["", ""])
                current[index][part] = value
                on_change(current)

            return update

        return tuple(
            (
                mo.ui.text(
                    value="" if idx >= len(items) else str(items[idx][0]),
                    label="",
                    placeholder=key_placeholder,
                    full_width=True,
                    on_change=update_at(idx, 0),
                ),
                mo.ui.text(
                    value=item_value(idx),
                    label="",
                    placeholder=value_placeholder,
                    full_width=True,
                    on_change=update_at(idx, 1),
                ),
            )
            for idx in range(count)
        )

    def dynamic_dict_entries(
        entry_widgets: tuple[tuple[object, object], ...],
        count_widget: object,
        tooltip: str | None = None,
    ):
        rows = [
            mo.hstack(
                [key_widget, value_widget],
                widths=[0.45, 0.55],
                gap=0.5,
                align="center",
            )
            for key_widget, value_widget in entry_widgets
        ]
        return dynamic_array(mo.vstack(rows, gap=0.15), count_widget, tooltip)

    def dynamic_entries(
        entry_widgets: tuple[object, ...],
        count_widget: object,
        *,
        tooltip: str | None = None,
        entry_tooltip: str | None = None,
    ):
        entries = [with_help(widget, entry_tooltip) for widget in entry_widgets]
        entries_widget = mo.vstack(entries, gap=0.15)
        return dynamic_array(entries_widget, count_widget, tooltip)

    def dynamic_text_array(
        values: object,
        count: int,
        *,
        label: str,
        on_change=None,
        options: tuple[str, ...] | None = None,
    ):
        items = list(values) if isinstance(values, list | tuple) else []
        full_width = label != "protocols"

        def update_at(index: int):
            def update(value: str):
                if on_change is None:
                    return
                current = [str(item) for item in items]
                while len(current) < count:
                    current.append("")
                current[index] = value
                on_change(current)

            return update

        return tuple(
            (
                mo.ui.dropdown(
                    options=options,
                    value=str(items[idx])
                    if idx < len(items) and str(items[idx]) in options
                    else options[0],
                    label="",
                    on_change=update_at(idx),
                )
                if options is not None
                else mo.ui.text(
                    value="" if idx >= len(items) else str(items[idx]),
                    label="",
                    placeholder="value",
                    full_width=full_width,
                    on_change=update_at(idx),
                )
            )
            for idx in range(count)
        )

    def dynamic_manifest_array(values: object, count: int, on_change=None):
        items = (
            list(values.items())
            if isinstance(values, dict)
            else list(values)
            if isinstance(values, list | tuple)
            else []
        )

        def item_value(index: int) -> str:
            if index >= len(items):
                return ""
            if isinstance(items[index], str):
                return str(items[index])
            return f"{json.dumps(str(items[index][0]))}: {json.dumps(str(items[index][1]))}"

        def update_at(index: int):
            def update(value: str):
                if on_change is None:
                    return
                current = [item_value(idx) for idx in range(count)]
                current[index] = value
                on_change(current)

            return update

        return tuple(
            mo.ui.text(
                value=item_value(idx),
                label="",
                placeholder='"type=.../manifest.parquet": "git-commit-prefix"',
                full_width=True,
                on_change=update_at(idx),
            )
            for idx in range(count)
        )

    def scaling_rule_widgets(columns: object, values: object, on_change=None):
        selected_columns = tuple(
            dict.fromkeys(str(column).strip() for column in columns if str(column).strip())
        )
        configured = (
            tuple(item for item in values if isinstance(item, dict))
            if isinstance(values, list | tuple)
            else ()
        )
        initial = list(scaling_drafts(selected_columns, configured))

        def update_at(index: int, field: str):
            def update(value: object):
                if on_change is None:
                    return
                current = [dict(item) for item in initial]
                current[index][field] = value
                on_change(current)

            return update

        return tuple(
            {
                "column": rule["column"],
                "input_min": mo.ui.text(
                    value=str(rule["input_min"]),
                    label="input min",
                    on_change=update_at(index, "input_min"),
                ),
                "input_max": mo.ui.text(
                    value=str(rule["input_max"]),
                    label="input max",
                    on_change=update_at(index, "input_max"),
                ),
                "output_min": mo.ui.text(
                    value=str(rule["output_min"]),
                    label="output min",
                    on_change=update_at(index, "output_min"),
                ),
                "output_max": mo.ui.text(
                    value=str(rule["output_max"]),
                    label="output max",
                    on_change=update_at(index, "output_max"),
                ),
                "transform": mo.ui.dropdown(
                    options=("linear", "log1p"),
                    value=str(rule["transform"])
                    if str(rule["transform"]) in {"linear", "log1p"}
                    else "linear",
                    label="transform",
                    on_change=update_at(index, "transform"),
                ),
                "clip": mo.ui.checkbox(
                    value=bool(rule["clip"]),
                    label="clip",
                    on_change=update_at(index, "clip"),
                ),
            }
            for index, rule in enumerate(initial)
        )

    return (
        dict_entry_widgets,
        dynamic_dict_entries,
        dynamic_manifest_array,
        dynamic_text_array,
        info_icon,
        scaling_rule_widgets,
        tree_array_rows,
        with_help,
    )


@app.cell
def _(get_form_drafts, load_path, loaded_config, path_value, set_form_drafts):
    form_drafts = get_form_drafts()
    current_load_path = str(load_path.value)
    form_values = form_drafts.get(current_load_path, {})

    def _remember_count(path: str):
        def update(value: object):
            drafts = dict(get_form_drafts())
            draft = dict(drafts.get(current_load_path, {}))
            draft[f"__count__.{path}"] = int(value)
            drafts[current_load_path] = draft
            set_form_drafts(drafts)

        return update

    def count_widget(path: str, default: object, *, required: bool = False):
        value = (
            form_values[path] if path in form_values else path_value(loaded_config, path, default)
        )
        count = form_values.get(
            f"__count__.{path}",
            len(value) if isinstance(value, dict | list | tuple) else int(required),
        )
        minimum = int(required)
        return mo.ui.number(
            value=max(minimum, int(count)),
            start=minimum,
            step=1,
            label="",
            on_change=_remember_count(path),
        )

    data_manifest_count = count_widget("data.manifest_paths", {}, required=True)
    data_protocol_count = count_widget("data.protocols", [], required=True)
    data_input_column_count = count_widget("data.input_columns", [], required=True)
    data_target_column_count = count_widget("data.target_columns", [], required=True)
    data_feedback_column_count = count_widget("data.feedback_columns", [])
    model_layer_count = count_widget(
        "model.layers",
        [
            {"kind": "reduce", "mode": "sum_pool"},
            {"kind": "attention"},
            {"kind": "ffn"},
        ],
        required=True,
    )
    model_head_layer_count = count_widget("model.head_layers", [{"kind": "ffn"}])
    train_masked_suffix_channel_count = count_widget("train.masked_suffix.channels", [])
    validation_group_by_count = count_widget(
        "validation.split.group_by",
        ["dataset id", "cell id", "cycle index"],
        required=True,
    )
    validation_split_group_count = count_widget("validation.split.groups", [])
    validation_rollout_extension_input_value_count = count_widget(
        "validation.rollout_extension.input_values", {}
    )
    logging_wandb_tag_count = count_widget("logging.wandb.tags", [])
    checkpoint_monitor_count = count_widget("checkpoint.monitors", [])
    return (
        checkpoint_monitor_count,
        data_feedback_column_count,
        data_input_column_count,
        data_manifest_count,
        data_protocol_count,
        data_target_column_count,
        logging_wandb_tag_count,
        model_head_layer_count,
        model_layer_count,
        train_masked_suffix_channel_count,
        validation_group_by_count,
        validation_rollout_extension_input_value_count,
        validation_split_group_count,
    )


@app.cell
def _(
    get_form_drafts,
    load_path,
    loaded_config,
    path_value,
    set_form_drafts,
    validation_split_group_count,
):
    _form_drafts = get_form_drafts()
    _current_load_path = str(load_path.value)
    _form_values = _form_drafts.get(_current_load_path, {})
    groups = (
        _form_values["validation.split.groups"]
        if "validation.split.groups" in _form_values
        else path_value(loaded_config, "validation.split.groups", [])
    )
    group_items = list(groups) if isinstance(groups, list | tuple) else []

    def group_value(index: int) -> dict[str, object]:
        if index >= len(group_items) or not isinstance(group_items[index], dict):
            return {}
        return group_items[index]

    def count_for(value: object, *, required: bool = False) -> int:
        minimum = int(required)
        return max(minimum, len(value)) if isinstance(value, dict | list | tuple) else minimum

    def cached_count(key: str, index: int, default: int) -> int:
        minimum = 1 if key == "__validation_split_group_match_counts" else 0
        counts = _form_values.get(key, [])
        if isinstance(counts, list | tuple) and index < len(counts):
            return max(minimum, int(counts[index]))
        return default

    def remember_count(key: str, index: int):
        def update(value: object):
            minimum = 1 if key == "__validation_split_group_match_counts" else 0
            drafts = dict(get_form_drafts())
            current = dict(drafts.get(_current_load_path, {}))
            counts = list(current.get(key, []))
            while len(counts) <= index:
                counts.append(1)
            counts[index] = max(minimum, int(value))
            current[key] = counts
            drafts[_current_load_path] = current
            set_form_drafts(drafts)

        return update

    validation_split_group_match_counts = tuple(
        mo.ui.number(
            value=cached_count(
                "__validation_split_group_match_counts",
                idx,
                count_for(group_value(idx).get("match", {}), required=True),
            ),
            start=1,
            step=1,
            label="",
            on_change=remember_count("__validation_split_group_match_counts", idx),
        )
        for idx in range(int(validation_split_group_count.value))
    )
    validation_split_group_offset_counts = tuple(
        mo.ui.number(
            value=cached_count(
                "__validation_split_group_offset_counts",
                idx,
                count_for(group_value(idx).get("rollout_start_offsets", [])),
            ),
            start=0,
            step=1,
            label="",
            on_change=remember_count("__validation_split_group_offset_counts", idx),
        )
        for idx in range(int(validation_split_group_count.value))
    )
    return (
        validation_split_group_match_counts,
        validation_split_group_offset_counts,
    )


@app.cell
def _(
    data_feedback_column_count,
    data_input_column_count,
    data_manifest_count,
    data_protocol_count,
    data_target_column_count,
    dict_entry_widgets,
    dynamic_manifest_array,
    dynamic_text_array,
    get_form_drafts,
    load_path,
    loaded_config,
    model_head_layer_count,
    model_layer_count,
    nullable_bool_value,
    path_value,
    set_form_drafts,
    set_layer_kind_change,
    train_masked_suffix_channel_count,
    validation_group_by_count,
    validation_rollout_extension_input_value_count,
    validation_split_group_count,
    validation_split_group_match_counts,
    validation_split_group_offset_counts,
):
    _form_drafts = get_form_drafts()
    _current_load_path = str(load_path.value)
    _form_values = _form_drafts.get(_current_load_path, {})

    def _source_value(path: str, default: object):
        if path in _form_values:
            value = _form_values[path]
        else:
            value = path_value(loaded_config, path, default)
        if isinstance(default, bool):
            return value if isinstance(value, bool) else default
        if isinstance(default, int):
            return value if isinstance(value, int) and not isinstance(value, bool) else default
        if isinstance(default, float):
            return (
                value if isinstance(value, int | float) and not isinstance(value, bool) else default
            )
        if isinstance(default, str):
            return value if isinstance(value, str) else default
        if isinstance(default, list):
            return value if isinstance(value, list | tuple) else default
        if isinstance(default, dict):
            return value if isinstance(value, dict | list | tuple) else default
        return value

    def _choice(path: str, default: str, options: tuple[str, ...]) -> str:
        value = str(_source_value(path, default))
        return value if value in options else default

    def _remember(path: str):
        def update(value):
            drafts = dict(get_form_drafts())
            current = dict(drafts.get(_current_load_path, {}))
            current[path] = value
            drafts[_current_load_path] = current
            set_form_drafts(drafts)

        return update

    data_manifest_paths = dynamic_manifest_array(
        _source_value("data.manifest_paths", {}),
        int(data_manifest_count.value),
        on_change=_remember("data.manifest_paths"),
    )
    data_protocols = dynamic_text_array(
        _source_value("data.protocols", []),
        int(data_protocol_count.value),
        label="protocols",
        on_change=_remember("data.protocols"),
        options=("cycling", "HPPC", "RPT", "EIS"),
    )
    data_protocol_mode = mo.ui.dropdown(
        options=("available", "strict"),
        value=_choice("data.protocol_mode", "available", ("available", "strict")),
        label="",
        on_change=_remember("data.protocol_mode"),
    )
    data_store_root = mo.ui.text(
        value=""
        if _source_value("data.store_root", None) is None
        else str(_source_value("data.store_root", "")),
        label="",
        full_width=True,
        on_change=_remember("data.store_root"),
    )
    data_input_columns = dynamic_text_array(
        _source_value("data.input_columns", []),
        int(data_input_column_count.value),
        label="input_columns",
        on_change=_remember("data.input_columns"),
    )
    data_target_columns = dynamic_text_array(
        _source_value("data.target_columns", []),
        int(data_target_column_count.value),
        label="target_columns",
        on_change=_remember("data.target_columns"),
    )
    data_feedback_columns = dynamic_text_array(
        _source_value("data.feedback_columns", []),
        int(data_feedback_column_count.value),
        label="feedback_columns",
        on_change=_remember("data.feedback_columns"),
    )
    loader_batch_size = mo.ui.number(
        value=int(_source_value("loader.batch_size", 32)),
        start=1,
        step=1,
        label="",
        on_change=_remember("loader.batch_size"),
    )
    loader_seq_len = mo.ui.number(
        value=int(_source_value("loader.seq_len", 1024)),
        start=1,
        step=1,
        label="",
        on_change=_remember("loader.seq_len"),
    )
    loader_strategy = mo.ui.dropdown(
        options=("sequential", "shuffled_protocol_groups"),
        value=_choice(
            "loader.strategy",
            "shuffled_protocol_groups",
            ("sequential", "shuffled_protocol_groups"),
        ),
        label="",
        on_change=_remember("loader.strategy"),
    )
    loader_stateful_n_windows = mo.ui.number(
        value=int(_source_value("loader.stateful_n_windows", 1)),
        start=-1,
        step=1,
        label="",
        on_change=_remember("loader.stateful_n_windows"),
    )
    loader_cross_protocol_state_carry = mo.ui.dropdown(
        options=("null", "chain"),
        value="null"
        if _source_value("loader.cross_protocol_state_carry", None) is None
        else _choice("loader.cross_protocol_state_carry", "chain", ("null", "chain")),
        label="",
        on_change=_remember("loader.cross_protocol_state_carry"),
    )
    loader_data_access = mo.ui.dropdown(
        options=("windowed", "full_in_mem"),
        value=_choice("loader.data_access", "windowed", ("windowed", "full_in_mem")),
        label="",
        on_change=_remember("loader.data_access"),
    )
    loader_num_workers = mo.ui.number(
        value=int(_source_value("loader.num_workers", 0)),
        start=0,
        step=1,
        label="",
        on_change=_remember("loader.num_workers"),
    )
    loader_prefetch_to_device = mo.ui.checkbox(
        value=bool(_source_value("loader.prefetch_to_device", False)),
        label="",
        on_change=_remember("loader.prefetch_to_device"),
    )

    def model_value(path: str, default):
        return _source_value(f"model.{path}", default)

    def nullable_text(value: object):
        return "" if value is None else str(value)

    def model_layer_widgets(path: str, count: int):
        default_layers = (
            [
                {"kind": "reduce", "mode": "sum_pool"},
                {"kind": "attention"},
                {"kind": "ffn"},
            ]
            if path == "layers"
            else [{"kind": "ffn"}]
        )
        values = model_value(path, default_layers)
        layers = list(values) if isinstance(values, list | tuple) else []

        def layer_value(index: int, key: str, default):
            if index >= len(layers) or not isinstance(layers[index], dict):
                return default
            return layers[index].get(key, default)

        def layer_choice(index: int, key: str, default: str, options: tuple[str, ...]):
            value = str(layer_value(index, key, default))
            return value if value in options else default

        def residual_value(index: int) -> str:
            value = layer_value(index, "residual", None)
            if isinstance(value, str) and value in {
                "inherit/null",
                "boolean:true",
                "boolean:false",
                "object:standard",
                "object:none",
            }:
                return str(value)
            if isinstance(value, dict):
                return f"object:{value.get('kind', 'standard')}"
            if value is True:
                return "boolean:true"
            if value is False:
                return "boolean:false"
            return "inherit/null"

        def remember_layer(index: int, key: str):
            def update(value: object):
                current_layers = [
                    dict(layer) if isinstance(layer, dict) else {} for layer in layers
                ]
                while len(current_layers) < count:
                    current_layers.append({})
                current_layers[index][key] = value
                _remember(f"model.{path}")(current_layers)

            return update

        def notify_layer_kind_change(_value: object) -> None:
            set_layer_kind_change(object())

        return tuple(
            {
                "kind": mo.ui.dropdown(
                    options=("reduce", "attention", "ffn", "mamba"),
                    value=layer_choice(
                        idx,
                        "kind",
                        "ffn",
                        ("reduce", "attention", "ffn", "mamba"),
                    ),
                    label="",
                    on_change=lambda value, index=idx: (
                        remember_layer(index, "kind")(value),
                        notify_layer_kind_change(value),
                    ),
                ),
                "residual": mo.ui.dropdown(
                    options=(
                        "inherit/null",
                        "boolean:true",
                        "boolean:false",
                        "object:standard",
                        "object:none",
                    ),
                    value=residual_value(idx),
                    label="",
                    on_change=remember_layer(idx, "residual"),
                ),
                "bias": mo.ui.dropdown(
                    options=("inherit/null", "true", "false"),
                    value=nullable_bool_value(layer_value(idx, "bias", None)),
                    label="",
                    on_change=remember_layer(idx, "bias"),
                ),
                "d_state": mo.ui.text(
                    value=nullable_text(layer_value(idx, "d_state", None)),
                    label="",
                    on_change=remember_layer(idx, "d_state"),
                ),
                "expand": mo.ui.text(
                    value=nullable_text(layer_value(idx, "expand", None)),
                    label="",
                    on_change=remember_layer(idx, "expand"),
                ),
                "headdim": mo.ui.text(
                    value=nullable_text(layer_value(idx, "headdim", None)),
                    label="",
                    on_change=remember_layer(idx, "headdim"),
                ),
                "ngroups": mo.ui.text(
                    value=nullable_text(layer_value(idx, "ngroups", None)),
                    label="",
                    on_change=remember_layer(idx, "ngroups"),
                ),
                "is_mimo": mo.ui.dropdown(
                    options=("inherit/null", "true", "false"),
                    value=nullable_bool_value(layer_value(idx, "is_mimo", None)),
                    label="",
                    on_change=remember_layer(idx, "is_mimo"),
                ),
                "mimo_rank": mo.ui.text(
                    value=nullable_text(layer_value(idx, "mimo_rank", None)),
                    label="",
                    on_change=remember_layer(idx, "mimo_rank"),
                ),
                "chunk_size": mo.ui.text(
                    value=nullable_text(layer_value(idx, "chunk_size", None)),
                    label="",
                    on_change=remember_layer(idx, "chunk_size"),
                ),
            }
            for idx in range(count)
        )

    def model_change(path: str):
        return _remember(f"model.{path}")

    model_d_model = mo.ui.number(
        value=int(model_value("d_model", 256)),
        start=1,
        step=1,
        label="",
        on_change=model_change("d_model"),
    )
    model_n_heads = mo.ui.number(
        value=int(model_value("n_heads", 8)),
        start=1,
        step=1,
        label="",
        on_change=model_change("n_heads"),
    )
    model_mlp_ratio = mo.ui.number(
        value=float(model_value("mlp_ratio", 4.0)),
        start=0.01,
        step=0.25,
        label="",
        on_change=model_change("mlp_ratio"),
    )
    model_dropout = mo.ui.number(
        value=float(model_value("dropout", 0.0)),
        start=0.0,
        stop=0.999999,
        step=0.01,
        label="",
        on_change=model_change("dropout"),
    )
    model_bias = mo.ui.checkbox(
        value=bool(model_value("bias", False)),
        label="",
        on_change=model_change("bias"),
    )
    model_norm = mo.ui.dropdown(
        options=("rmsnorm",),
        value=_choice("model.norm", "rmsnorm", ("rmsnorm",)),
        label="",
        on_change=model_change("norm"),
    )
    model_causal_attention = mo.ui.checkbox(
        value=bool(model_value("causal_attention", True)),
        label="",
        on_change=model_change("causal_attention"),
    )
    model_num_bins = mo.ui.number(
        value=int(model_value("num_bins", 64)),
        start=2,
        step=1,
        label="",
        on_change=model_change("num_bins"),
    )
    model_input_sigma = mo.ui.number(
        value=float(model_value("input_sigma", 0.0)),
        start=0.0,
        step=0.01,
        label="",
        on_change=model_change("input_sigma"),
    )
    model_output_sigma = mo.ui.number(
        value=float(model_value("output_sigma", 0.0)),
        start=0.0,
        step=0.01,
        label="",
        on_change=model_change("output_sigma"),
    )
    model_feedback_mode = mo.ui.dropdown(
        options=("probabilities", "decoded_scalar"),
        value=_choice(
            "model.feedback_mode",
            "probabilities",
            ("probabilities", "decoded_scalar"),
        ),
        label="",
        on_change=model_change("feedback_mode"),
    )
    model_mamba_d_state = mo.ui.number(
        value=int(model_value("mamba.d_state", 128)),
        start=1,
        step=1,
        label="",
        on_change=model_change("mamba.d_state"),
    )
    model_mamba_expand = mo.ui.number(
        value=int(model_value("mamba.expand", 2)),
        start=1,
        step=1,
        label="",
        on_change=model_change("mamba.expand"),
    )
    model_mamba_headdim = mo.ui.number(
        value=int(model_value("mamba.headdim", 64)),
        start=1,
        step=1,
        label="",
        on_change=model_change("mamba.headdim"),
    )
    model_mamba_ngroups = mo.ui.number(
        value=int(model_value("mamba.ngroups", 1)),
        start=1,
        step=1,
        label="",
        on_change=model_change("mamba.ngroups"),
    )
    model_mamba_is_mimo = mo.ui.checkbox(
        value=bool(model_value("mamba.is_mimo", False)),
        label="",
        on_change=model_change("mamba.is_mimo"),
    )
    model_mamba_mimo_rank = mo.ui.number(
        value=int(model_value("mamba.mimo_rank", 1)),
        start=1,
        step=1,
        label="",
        on_change=model_change("mamba.mimo_rank"),
    )
    model_mamba_chunk_size = mo.ui.number(
        value=int(model_value("mamba.chunk_size", 64)),
        start=1,
        step=1,
        label="",
        on_change=model_change("mamba.chunk_size"),
    )
    model_layers = model_layer_widgets("layers", int(model_layer_count.value))
    model_head_layers = model_layer_widgets("head_layers", int(model_head_layer_count.value))
    model_output_parameterization = mo.ui.dropdown(
        options=("shared",),
        value=_choice("model.output.parameterization", "shared", ("shared",)),
        label="",
        on_change=model_change("output.parameterization"),
    )
    train_epochs = mo.ui.number(
        value=float(_source_value("train.epochs", 1.0)),
        start=0.25,
        step=0.25,
        label="",
        on_change=_remember("train.epochs"),
    )
    train_log_per_epoch = mo.ui.number(
        value=int(_source_value("train.log_per_epoch", 10)),
        start=1,
        step=1,
        label="",
        on_change=_remember("train.log_per_epoch"),
    )
    train_max_steps = mo.ui.text(
        value=""
        if _source_value("train.max_steps", None) is None
        else str(_source_value("train.max_steps", "")),
        label="",
        on_change=_remember("train.max_steps"),
    )
    train_log_every_steps = mo.ui.text(
        value=""
        if _source_value("train.log_every_steps", None) is None
        else str(_source_value("train.log_every_steps", "")),
        label="",
        on_change=_remember("train.log_every_steps"),
    )
    train_validate_every_steps = mo.ui.text(
        value=""
        if _source_value("train.validate_every_steps", None) is None
        else str(_source_value("train.validate_every_steps", "")),
        label="",
        on_change=_remember("train.validate_every_steps"),
    )
    train_validate_per_epoch = mo.ui.number(
        value=int(_source_value("train.validate_per_epoch", 1)),
        start=0,
        step=1,
        label="",
        on_change=_remember("train.validate_per_epoch"),
    )
    train_grad_clip_norm = mo.ui.number(
        value=float(_source_value("train.grad_clip_norm", 1.0)),
        start=0.1,
        step=0.1,
        label="",
        on_change=_remember("train.grad_clip_norm"),
    )
    train_masked_suffix_enabled = mo.ui.checkbox(
        value=bool(_source_value("train.masked_suffix.enabled", True)),
        label="",
        on_change=_remember("train.masked_suffix.enabled"),
    )
    train_masked_suffix_channels = dynamic_text_array(
        _source_value("train.masked_suffix.channels", []),
        int(train_masked_suffix_channel_count.value),
        label="channels",
        on_change=_remember("train.masked_suffix.channels"),
    )
    train_masked_suffix_suffix_steps = mo.ui.number(
        value=int(_source_value("train.masked_suffix.suffix_steps", 128)),
        start=1,
        step=1,
        label="",
        on_change=_remember("train.masked_suffix.suffix_steps"),
    )
    train_masked_suffix_loss_on_masked_only = mo.ui.checkbox(
        value=bool(_source_value("train.masked_suffix.loss_on_masked_only", True)),
        label="",
        on_change=_remember("train.masked_suffix.loss_on_masked_only"),
    )
    train_masked_suffix_carry_mamba_state = mo.ui.checkbox(
        value=bool(_source_value("train.masked_suffix.carry_mamba_state", True)),
        label="",
        on_change=_remember("train.masked_suffix.carry_mamba_state"),
    )
    train_masked_suffix_detach_between_windows = mo.ui.checkbox(
        value=bool(_source_value("train.masked_suffix.detach_between_windows", True)),
        label="",
        on_change=_remember("train.masked_suffix.detach_between_windows"),
    )
    train_masked_suffix_roll_forward_steps = mo.ui.number(
        value=int(_source_value("train.masked_suffix.roll_forward_steps", 0)),
        start=0,
        step=1,
        label="",
        on_change=_remember("train.masked_suffix.roll_forward_steps"),
    )
    validation_split_strategy = mo.ui.dropdown(
        options=("sample", "provide", "merge"),
        value=_choice(
            "validation.split.strategy",
            "sample",
            ("sample", "provide", "merge"),
        ),
        label="",
        on_change=_remember("validation.split.strategy"),
    )
    validation_split_fraction = mo.ui.number(
        value=float(_source_value("validation.split.fraction", 0.2)),
        start=0.0,
        stop=0.999999,
        step=0.05,
        label="",
        on_change=_remember("validation.split.fraction"),
    )
    validation_split_group_by = dynamic_text_array(
        _source_value("validation.split.group_by", []),
        int(validation_group_by_count.value),
        label="group_by",
        on_change=_remember("validation.split.group_by"),
    )
    raw_validation_split_groups = _source_value("validation.split.groups", [])
    validation_split_group_items = (
        list(raw_validation_split_groups)
        if isinstance(raw_validation_split_groups, list | tuple)
        else []
    )

    def _remember_group_field(index: int, field: str):
        def update(value):
            drafts = dict(get_form_drafts())
            current = dict(drafts.get(_current_load_path, {}))
            groups = list(current.get("validation.split.groups", validation_split_group_items))
            while len(groups) <= index:
                groups.append({})
            group = dict(groups[index]) if isinstance(groups[index], dict) else {}
            if field == "match":
                group[field] = {
                    str(key).strip(): item_value for key, item_value in value if str(key).strip()
                }
            else:
                group[field] = value
            groups[index] = group
            current["validation.split.groups"] = groups
            drafts[_current_load_path] = current
            set_form_drafts(drafts)

        return update

    validation_split_groups = tuple(
        {
            "match": dict_entry_widgets(
                validation_split_group_items[idx].get("match", {})
                if idx < len(validation_split_group_items)
                and isinstance(validation_split_group_items[idx], dict)
                else {},
                int(validation_split_group_match_counts[idx].value),
                key_placeholder="match column",
                value_placeholder="match value",
                on_change=_remember_group_field(idx, "match"),
            ),
            "match_count": validation_split_group_match_counts[idx],
            "rollout_start_offsets": dynamic_text_array(
                validation_split_group_items[idx].get("rollout_start_offsets", [])
                if idx < len(validation_split_group_items)
                and isinstance(validation_split_group_items[idx], dict)
                else [],
                int(validation_split_group_offset_counts[idx].value),
                label="rollout_start_offsets",
                on_change=_remember_group_field(idx, "rollout_start_offsets"),
            ),
            "rollout_start_offset_count": validation_split_group_offset_counts[idx],
        }
        for idx in range(int(validation_split_group_count.value))
    )
    validation_max_tf_batches = mo.ui.number(
        value=int(_source_value("validation.max_tf_batches", 1)),
        start=0,
        step=1,
        label="",
        on_change=_remember("validation.max_tf_batches"),
    )
    validation_rollout_steps = mo.ui.number(
        value=int(_source_value("validation.rollout_steps", 0)),
        start=0,
        step=1,
        label="",
        on_change=_remember("validation.rollout_steps"),
    )
    validation_log_rollout_plots = mo.ui.checkbox(
        value=bool(_source_value("validation.log_rollout_plots", True)),
        label="",
        on_change=_remember("validation.log_rollout_plots"),
    )
    validation_masked_suffix_enabled = mo.ui.dropdown(
        options=("inherit/null", "true", "false"),
        value=nullable_bool_value(_source_value("validation.masked_suffix.enabled", None)),
        label="",
        on_change=_remember("validation.masked_suffix.enabled"),
    )
    validation_masked_suffix_suffix_steps = mo.ui.text(
        value=""
        if _source_value("validation.masked_suffix.suffix_steps", None) is None
        else str(_source_value("validation.masked_suffix.suffix_steps", "")),
        label="",
        on_change=_remember("validation.masked_suffix.suffix_steps"),
    )
    validation_masked_suffix_carry_mamba_state = mo.ui.dropdown(
        options=("inherit/null", "true", "false"),
        value=nullable_bool_value(
            _source_value("validation.masked_suffix.carry_mamba_state", None)
        ),
        label="",
        on_change=_remember("validation.masked_suffix.carry_mamba_state"),
    )
    validation_rollout_extension_enabled = mo.ui.checkbox(
        value=bool(_source_value("validation.rollout_extension.enabled", False)),
        label="",
        on_change=_remember("validation.rollout_extension.enabled"),
    )
    validation_rollout_extension_steps = mo.ui.number(
        value=int(_source_value("validation.rollout_extension.steps", 0)),
        start=0,
        step=1,
        label="",
        on_change=_remember("validation.rollout_extension.steps"),
    )
    validation_rollout_extension_input_values = dict_entry_widgets(
        _source_value("validation.rollout_extension.input_values", {}),
        int(validation_rollout_extension_input_value_count.value),
        key_placeholder="input column",
        value_placeholder="float value",
        on_change=_remember("validation.rollout_extension.input_values"),
    )
    return (
        data_feedback_columns,
        data_input_columns,
        data_manifest_paths,
        data_protocol_mode,
        data_protocols,
        data_store_root,
        data_target_columns,
        loader_batch_size,
        loader_cross_protocol_state_carry,
        loader_data_access,
        loader_num_workers,
        loader_prefetch_to_device,
        loader_seq_len,
        loader_stateful_n_windows,
        loader_strategy,
        model_bias,
        model_causal_attention,
        model_d_model,
        model_dropout,
        model_feedback_mode,
        model_head_layers,
        model_input_sigma,
        model_layers,
        model_mamba_chunk_size,
        model_mamba_d_state,
        model_mamba_expand,
        model_mamba_headdim,
        model_mamba_is_mimo,
        model_mamba_mimo_rank,
        model_mamba_ngroups,
        model_mlp_ratio,
        model_n_heads,
        model_norm,
        model_num_bins,
        model_output_parameterization,
        model_output_sigma,
        train_epochs,
        train_grad_clip_norm,
        train_log_every_steps,
        train_log_per_epoch,
        train_masked_suffix_carry_mamba_state,
        train_masked_suffix_channels,
        train_masked_suffix_detach_between_windows,
        train_masked_suffix_enabled,
        train_masked_suffix_loss_on_masked_only,
        train_masked_suffix_roll_forward_steps,
        train_masked_suffix_suffix_steps,
        train_max_steps,
        train_validate_every_steps,
        train_validate_per_epoch,
        validation_log_rollout_plots,
        validation_masked_suffix_carry_mamba_state,
        validation_masked_suffix_enabled,
        validation_masked_suffix_suffix_steps,
        validation_max_tf_batches,
        validation_rollout_extension_enabled,
        validation_rollout_extension_input_values,
        validation_rollout_extension_steps,
        validation_rollout_steps,
        validation_split_fraction,
        validation_split_group_by,
        validation_split_groups,
        validation_split_strategy,
    )


@app.cell
def _(
    data_input_columns,
    data_target_columns,
    get_form_drafts,
    load_path,
    loaded_config,
    path_value,
    scaling_rule_widgets,
    set_form_drafts,
):
    _scaling_load_path = str(load_path.value)
    _scaling_form_drafts = get_form_drafts()
    _scaling_form_values = _scaling_form_drafts.get(_scaling_load_path, {})
    _configured_scaling = (
        _scaling_form_values["data.scaling"]
        if "data.scaling" in _scaling_form_values
        else path_value(loaded_config, "data.scaling", [])
    )
    _scaling_columns = tuple(
        dict.fromkeys(
            str(widget.value).strip()
            for widget in (*data_input_columns, *data_target_columns)
            if str(widget.value).strip()
        )
    )

    def _remember_scaling(value):
        drafts = dict(get_form_drafts())
        current = dict(drafts.get(_scaling_load_path, {}))
        current["data.scaling"] = value
        drafts[_scaling_load_path] = current
        set_form_drafts(drafts)

    data_scaling = scaling_rule_widgets(
        _scaling_columns,
        _configured_scaling,
        on_change=_remember_scaling,
    )
    return (data_scaling,)


@app.cell
def _(
    checkpoint_monitor_count,
    dynamic_text_array,
    get_form_drafts,
    load_path,
    loaded_config,
    logging_wandb_tag_count,
    path_value,
    set_form_drafts,
):
    _form_drafts = get_form_drafts()
    _current_load_path = str(load_path.value)
    _form_values = _form_drafts.get(_current_load_path, {})

    def _source_value(path: str, default: object):
        if path in _form_values:
            value = _form_values[path]
        else:
            value = path_value(loaded_config, path, default)
        if isinstance(default, bool):
            return value if isinstance(value, bool) else default
        if isinstance(default, int):
            return value if isinstance(value, int) and not isinstance(value, bool) else default
        if isinstance(default, float):
            return (
                value if isinstance(value, int | float) and not isinstance(value, bool) else default
            )
        if isinstance(default, str):
            return value if isinstance(value, str) else default
        if isinstance(default, list):
            return value if isinstance(value, list | tuple) else default
        if isinstance(default, dict):
            return value if isinstance(value, dict | list | tuple) else default
        return value

    def _choice(path: str, default: str, options: tuple[str, ...]) -> str:
        value = str(_source_value(path, default))
        return value if value in options else default

    def _remember(path: str):
        def update(value):
            drafts = dict(get_form_drafts())
            current = dict(drafts.get(_current_load_path, {}))
            current[path] = value
            drafts[_current_load_path] = current
            set_form_drafts(drafts)

        return update

    optim_lr = mo.ui.number(
        value=float(_source_value("optim.lr", 1e-4)),
        start=0.0,
        step=1e-5,
        label="",
        on_change=_remember("optim.lr"),
    )
    optim_weight_decay = mo.ui.number(
        value=float(_source_value("optim.weight_decay", 0.01)),
        start=0.0,
        step=0.001,
        label="",
        on_change=_remember("optim.weight_decay"),
    )
    optim_beta1 = mo.ui.number(
        value=float(_source_value("optim.beta1", 0.9)),
        start=0.0,
        stop=1.0,
        step=0.01,
        label="",
        on_change=_remember("optim.beta1"),
    )
    optim_beta2 = mo.ui.number(
        value=float(_source_value("optim.beta2", 0.95)),
        start=0.0,
        stop=1.0,
        step=0.01,
        label="",
        on_change=_remember("optim.beta2"),
    )
    optim_eps = mo.ui.number(
        value=float(_source_value("optim.eps", 1e-8)),
        start=0.0,
        step=1e-8,
        label="",
        on_change=_remember("optim.eps"),
    )
    scheduler_kind = mo.ui.dropdown(
        options=("linear_warmup_cosine", "none"),
        value=_choice(
            "scheduler.kind",
            "linear_warmup_cosine",
            ("linear_warmup_cosine", "none"),
        ),
        label="",
        on_change=_remember("scheduler.kind"),
    )
    scheduler_warmup_ratio = mo.ui.number(
        value=float(_source_value("scheduler.warmup_ratio", 0.05)),
        start=0.0,
        stop=1.0,
        step=0.01,
        label="",
        on_change=_remember("scheduler.warmup_ratio"),
    )
    scheduler_min_lr_ratio = mo.ui.number(
        value=float(_source_value("scheduler.min_lr_ratio", 0.01)),
        start=0.0,
        stop=1.0,
        step=0.01,
        label="",
        on_change=_remember("scheduler.min_lr_ratio"),
    )
    run_device = mo.ui.text(
        value=str(_source_value("run.device", "cuda")),
        label="",
        on_change=_remember("run.device"),
    )
    run_seed = mo.ui.number(
        value=int(_source_value("run.seed", 69)),
        step=1,
        label="",
        on_change=_remember("run.seed"),
    )
    run_use_amp = mo.ui.checkbox(
        value=bool(_source_value("run.use_amp", True)),
        label="",
        on_change=_remember("run.use_amp"),
    )
    run_compile_model = mo.ui.checkbox(
        value=bool(_source_value("run.compile_model", False)),
        label="",
        on_change=_remember("run.compile_model"),
    )
    run_init_from = mo.ui.text(
        value=""
        if _source_value("run.init_from", None) is None
        else str(_source_value("run.init_from", "")),
        label="",
        full_width=True,
        on_change=_remember("run.init_from"),
    )
    run_output_dir = mo.ui.text(
        value=""
        if _source_value("run.output_dir", "outputs/runs") is None
        else str(_source_value("run.output_dir", "outputs/runs")),
        label="",
        full_width=True,
        on_change=_remember("run.output_dir"),
    )
    run_name = mo.ui.text(
        value="" if _source_value("run.name", None) is None else str(_source_value("run.name", "")),
        label="",
        on_change=_remember("run.name"),
    )
    logging_backend = mo.ui.dropdown(
        options=("stdout", "jsonl", "wandb"),
        value=_choice("logging.backend", "jsonl", ("stdout", "jsonl", "wandb")),
        label="",
        on_change=_remember("logging.backend"),
    )
    logging_mode = mo.ui.dropdown(
        options=("offline", "online"),
        value=_choice("logging.mode", "offline", ("offline", "online")),
        label="",
        on_change=_remember("logging.mode"),
    )
    logging_mirror_stdout = mo.ui.checkbox(
        value=bool(_source_value("logging.mirror_stdout", True)),
        label="",
        on_change=_remember("logging.mirror_stdout"),
    )
    logging_wandb_value = _source_value("logging.wandb", {})
    logging_wandb_value = logging_wandb_value if isinstance(logging_wandb_value, dict) else {}
    logging_wandb = {
        "project": mo.ui.text(
            value=""
            if _source_value("logging.wandb.project", None) is None
            else str(_source_value("logging.wandb.project", "")),
            label="",
            full_width=True,
            on_change=_remember("logging.wandb.project"),
        ),
        "entity": mo.ui.text(
            value=""
            if _source_value("logging.wandb.entity", None) is None
            else str(_source_value("logging.wandb.entity", "")),
            label="",
            full_width=True,
            on_change=_remember("logging.wandb.entity"),
        ),
        "group": mo.ui.text(
            value=""
            if _source_value("logging.wandb.group", None) is None
            else str(_source_value("logging.wandb.group", "")),
            label="",
            full_width=True,
            on_change=_remember("logging.wandb.group"),
        ),
        "name": mo.ui.text(
            value=""
            if _source_value("logging.wandb.name", None) is None
            else str(_source_value("logging.wandb.name", "")),
            label="",
            full_width=True,
            on_change=_remember("logging.wandb.name"),
        ),
        "tags": dynamic_text_array(
            _source_value("logging.wandb.tags", logging_wandb_value.get("tags", [])),
            int(logging_wandb_tag_count.value),
            label="tags",
            on_change=_remember("logging.wandb.tags"),
        ),
        "tags_count": logging_wandb_tag_count,
    }
    checkpoint_save_latest = mo.ui.checkbox(
        value=bool(_source_value("checkpoint.save_latest", False)),
        label="",
        on_change=_remember("checkpoint.save_latest"),
    )
    checkpoint_save_best = mo.ui.checkbox(
        value=bool(_source_value("checkpoint.save_best", False)),
        label="",
        on_change=_remember("checkpoint.save_best"),
    )
    checkpoint_save_final = mo.ui.checkbox(
        value=bool(_source_value("checkpoint.save_final", False)),
        label="",
        on_change=_remember("checkpoint.save_final"),
    )
    checkpoint_monitors = dynamic_text_array(
        _source_value("checkpoint.monitors", []),
        int(checkpoint_monitor_count.value),
        label="monitors",
        on_change=_remember("checkpoint.monitors"),
    )
    return (
        checkpoint_monitors,
        checkpoint_save_best,
        checkpoint_save_final,
        checkpoint_save_latest,
        logging_backend,
        logging_mirror_stdout,
        logging_mode,
        logging_wandb,
        optim_beta1,
        optim_beta2,
        optim_eps,
        optim_lr,
        optim_weight_decay,
        run_compile_model,
        run_device,
        run_init_from,
        run_name,
        run_output_dir,
        run_seed,
        run_use_amp,
        scheduler_kind,
        scheduler_min_lr_ratio,
        scheduler_warmup_ratio,
    )


@app.cell
def _(
    checkpoint_monitors,
    checkpoint_save_best,
    checkpoint_save_final,
    checkpoint_save_latest,
    data_feedback_columns,
    data_input_columns,
    data_manifest_paths,
    data_protocol_mode,
    data_protocols,
    data_scaling,
    data_store_root,
    data_target_columns,
    dict_entry_values,
    float_dict_entry_values,
    loader_batch_size,
    loader_cross_protocol_state_carry,
    loader_data_access,
    loader_num_workers,
    loader_prefetch_to_device,
    loader_seq_len,
    loader_stateful_n_windows,
    loader_strategy,
    logging_backend,
    logging_mirror_stdout,
    logging_mode,
    logging_wandb,
    manifest_values,
    model_bias,
    model_causal_attention,
    model_d_model,
    model_dropout,
    model_feedback_mode,
    model_head_layers,
    model_input_sigma,
    model_layers,
    model_mamba_chunk_size,
    model_mamba_d_state,
    model_mamba_expand,
    model_mamba_headdim,
    model_mamba_is_mimo,
    model_mamba_mimo_rank,
    model_mamba_ngroups,
    model_mlp_ratio,
    model_n_heads,
    model_norm,
    model_num_bins,
    model_output_parameterization,
    model_output_sigma,
    nonempty_values,
    optim_beta1,
    optim_beta2,
    optim_eps,
    optim_lr,
    optim_weight_decay,
    parse_nullable_bool,
    parse_nullable_int,
    parse_nullable_str,
    run_compile_model,
    run_device,
    run_init_from,
    run_name,
    run_output_dir,
    run_seed,
    run_use_amp,
    scaling_rule_values,
    scheduler_kind,
    scheduler_min_lr_ratio,
    scheduler_warmup_ratio,
    set_path,
    train_epochs,
    train_grad_clip_norm,
    train_log_every_steps,
    train_log_per_epoch,
    train_masked_suffix_carry_mamba_state,
    train_masked_suffix_channels,
    train_masked_suffix_detach_between_windows,
    train_masked_suffix_enabled,
    train_masked_suffix_loss_on_masked_only,
    train_masked_suffix_roll_forward_steps,
    train_masked_suffix_suffix_steps,
    train_max_steps,
    train_validate_every_steps,
    train_validate_per_epoch,
    validation_log_rollout_plots,
    validation_masked_suffix_carry_mamba_state,
    validation_masked_suffix_enabled,
    validation_masked_suffix_suffix_steps,
    validation_max_tf_batches,
    validation_rollout_extension_enabled,
    validation_rollout_extension_input_values,
    validation_rollout_extension_steps,
    validation_rollout_steps,
    validation_split_fraction,
    validation_split_group_by,
    validation_split_groups,
    validation_split_strategy,
):
    generated_config = {}
    build_errors = []

    manifest, manifest_errors = manifest_values(data_manifest_paths)
    build_errors.extend(manifest_errors)
    set_path(
        generated_config,
        "data.manifest_paths",
        manifest,
    )

    scaling, scaling_errors = scaling_rule_values(data_scaling)
    build_errors.extend(scaling_errors)
    set_path(generated_config, "data.scaling", scaling)

    try:
        stateful_n_windows = int(loader_stateful_n_windows.value)
        if stateful_n_windows == 0 or stateful_n_windows < -1:
            raise ValueError("loader.stateful_n_windows: must be -1 or a positive integer")
        set_path(
            generated_config,
            "data.protocols",
            nonempty_values(data_protocols),
        )
        set_path(
            generated_config,
            "data.protocol_mode",
            str(data_protocol_mode.value),
        )
        set_path(
            generated_config,
            "data.store_root",
            parse_nullable_str(data_store_root.value),
        )
        set_path(
            generated_config,
            "data.input_columns",
            nonempty_values(data_input_columns),
        )
        set_path(
            generated_config,
            "data.target_columns",
            nonempty_values(data_target_columns),
        )
        set_path(
            generated_config,
            "data.feedback_columns",
            nonempty_values(data_feedback_columns),
        )
        set_path(generated_config, "loader.batch_size", int(loader_batch_size.value))
        set_path(generated_config, "loader.seq_len", int(loader_seq_len.value))
        set_path(generated_config, "loader.strategy", str(loader_strategy.value))
        set_path(
            generated_config,
            "loader.stateful_n_windows",
            stateful_n_windows,
        )
        set_path(
            generated_config,
            "loader.cross_protocol_state_carry",
            None
            if loader_cross_protocol_state_carry.value == "null"
            else str(loader_cross_protocol_state_carry.value),
        )
        set_path(
            generated_config,
            "loader.data_access",
            str(loader_data_access.value),
        )
        set_path(
            generated_config,
            "loader.num_workers",
            int(loader_num_workers.value),
        )
        set_path(
            generated_config,
            "loader.prefetch_to_device",
            bool(loader_prefetch_to_device.value),
        )

        def build_layer(layer_widgets: dict[str, object], path: str) -> dict[str, object]:
            kind = str(layer_widgets["kind"].value)
            layer = {"kind": kind}
            residual = str(layer_widgets["residual"].value)
            if residual == "boolean:true":
                layer["residual"] = True
            elif residual == "boolean:false":
                layer["residual"] = False
            elif residual.startswith("object:"):
                layer["residual"] = {"kind": residual.removeprefix("object:")}
            if kind == "reduce":
                layer["mode"] = "sum_pool"
            if kind in {"attention", "ffn"}:
                bias = parse_nullable_bool(layer_widgets["bias"].value)
                if bias is not None:
                    layer["bias"] = bias
            if kind == "mamba":
                for key in ("d_state", "expand", "headdim", "ngroups", "mimo_rank", "chunk_size"):
                    try:
                        value = parse_nullable_int(layer_widgets[key].value)
                    except ValueError as exc:
                        raise ValueError(f"{path}.{key}: expected an integer") from exc
                    if value is not None:
                        if value <= 0:
                            raise ValueError(f"{path}.{key}: must be positive")
                        layer[key] = value
                is_mimo = parse_nullable_bool(layer_widgets["is_mimo"].value)
                if is_mimo is not None:
                    layer["is_mimo"] = is_mimo
            return layer

        set_path(
            generated_config,
            "model",
            {
                "d_model": int(model_d_model.value),
                "n_heads": int(model_n_heads.value),
                "mlp_ratio": float(model_mlp_ratio.value),
                "dropout": float(model_dropout.value),
                "bias": bool(model_bias.value),
                "norm": str(model_norm.value),
                "causal_attention": bool(model_causal_attention.value),
                "num_bins": int(model_num_bins.value),
                "input_sigma": float(model_input_sigma.value),
                "output_sigma": float(model_output_sigma.value),
                "feedback_mode": str(model_feedback_mode.value),
                "mamba": {
                    "d_state": int(model_mamba_d_state.value),
                    "expand": int(model_mamba_expand.value),
                    "headdim": int(model_mamba_headdim.value),
                    "ngroups": int(model_mamba_ngroups.value),
                    "is_mimo": bool(model_mamba_is_mimo.value),
                    "mimo_rank": int(model_mamba_mimo_rank.value),
                    "chunk_size": int(model_mamba_chunk_size.value),
                },
                "layers": [
                    build_layer(layer, f"model.layers[{index}]")
                    for index, layer in enumerate(model_layers)
                ],
                "head_layers": [
                    build_layer(layer, f"model.head_layers[{index}]")
                    for index, layer in enumerate(model_head_layers)
                ],
                "output": {"parameterization": str(model_output_parameterization.value)},
            },
        )
        set_path(generated_config, "train.epochs", float(train_epochs.value))
        set_path(generated_config, "train.loss", "categorical_ce")
        set_path(
            generated_config,
            "train.log_per_epoch",
            int(train_log_per_epoch.value),
        )
        for path, widget in (
            ("train.max_steps", train_max_steps),
            ("train.log_every_steps", train_log_every_steps),
            ("train.validate_every_steps", train_validate_every_steps),
        ):
            try:
                cadence = parse_nullable_int(widget.value)
            except ValueError as exc:
                raise ValueError(f"{path}: expected an integer or blank") from exc
            if cadence is not None and cadence <= 0:
                raise ValueError(f"{path}: must be positive when set")
            set_path(generated_config, path, cadence)
        set_path(
            generated_config,
            "train.validate_per_epoch",
            int(train_validate_per_epoch.value),
        )
        set_path(
            generated_config,
            "train.grad_clip_norm",
            float(train_grad_clip_norm.value),
        )
        set_path(
            generated_config,
            "train.masked_suffix.enabled",
            bool(train_masked_suffix_enabled.value),
        )
        set_path(
            generated_config,
            "train.masked_suffix.channels",
            nonempty_values(train_masked_suffix_channels),
        )
        set_path(
            generated_config,
            "train.masked_suffix.suffix_steps",
            int(train_masked_suffix_suffix_steps.value),
        )
        set_path(
            generated_config,
            "train.masked_suffix.loss_on_masked_only",
            bool(train_masked_suffix_loss_on_masked_only.value),
        )
        set_path(
            generated_config,
            "train.masked_suffix.carry_mamba_state",
            bool(train_masked_suffix_carry_mamba_state.value),
        )
        set_path(
            generated_config,
            "train.masked_suffix.detach_between_windows",
            bool(train_masked_suffix_detach_between_windows.value),
        )
        set_path(
            generated_config,
            "train.masked_suffix.roll_forward_steps",
            int(train_masked_suffix_roll_forward_steps.value),
        )
        set_path(
            generated_config,
            "validation.split.strategy",
            str(validation_split_strategy.value),
        )
        set_path(
            generated_config,
            "validation.split.fraction",
            float(validation_split_fraction.value),
        )
        set_path(
            generated_config,
            "validation.split.group_by",
            nonempty_values(validation_split_group_by),
        )
        validation_groups = []
        for group_index, group_widgets in enumerate(validation_split_groups):
            match, match_errors = dict_entry_values(
                f"validation.split.groups[{group_index}]",
                group_widgets["match"],
            )
            build_errors.extend(match_errors)
            offsets = [
                int(value) for value in nonempty_values(group_widgets["rollout_start_offsets"])
            ]
            if match or offsets:
                validation_groups.append({"match": match, "rollout_start_offsets": offsets})
        set_path(
            generated_config,
            "validation.split.groups",
            [] if validation_split_strategy.value == "sample" else validation_groups,
        )
        set_path(
            generated_config,
            "validation.max_tf_batches",
            int(validation_max_tf_batches.value),
        )
        set_path(
            generated_config,
            "validation.rollout_steps",
            int(validation_rollout_steps.value),
        )
        set_path(
            generated_config,
            "validation.log_rollout_plots",
            bool(validation_log_rollout_plots.value),
        )
        set_path(
            generated_config,
            "validation.masked_suffix.enabled",
            parse_nullable_bool(validation_masked_suffix_enabled.value),
        )
        set_path(
            generated_config,
            "validation.masked_suffix.suffix_steps",
            parse_nullable_int(validation_masked_suffix_suffix_steps.value),
        )
        set_path(
            generated_config,
            "validation.masked_suffix.carry_mamba_state",
            parse_nullable_bool(validation_masked_suffix_carry_mamba_state.value),
        )
        set_path(
            generated_config,
            "validation.rollout_extension.enabled",
            bool(validation_rollout_extension_enabled.value),
        )
        set_path(
            generated_config,
            "validation.rollout_extension.steps",
            int(validation_rollout_extension_steps.value),
        )
        set_path(
            generated_config,
            "validation.rollout_extension.input_values",
            float_dict_entry_values(
                "validation.rollout_extension.input_values",
                validation_rollout_extension_input_values,
            ),
        )
        set_path(generated_config, "optim.kind", "adamw")
        set_path(generated_config, "optim.lr", float(optim_lr.value))
        set_path(
            generated_config,
            "optim.weight_decay",
            float(optim_weight_decay.value),
        )
        set_path(generated_config, "optim.beta1", float(optim_beta1.value))
        set_path(generated_config, "optim.beta2", float(optim_beta2.value))
        set_path(generated_config, "optim.eps", float(optim_eps.value))
        set_path(generated_config, "scheduler.kind", str(scheduler_kind.value))
        set_path(
            generated_config,
            "scheduler.warmup_ratio",
            float(scheduler_warmup_ratio.value),
        )
        set_path(
            generated_config,
            "scheduler.min_lr_ratio",
            float(scheduler_min_lr_ratio.value),
        )
        device = str(run_device.value).strip()
        if not device:
            raise ValueError("run.device: must not be blank")
        set_path(generated_config, "run.device", device)
        set_path(generated_config, "run.seed", int(run_seed.value))
        set_path(generated_config, "run.use_amp", bool(run_use_amp.value))
        set_path(
            generated_config,
            "run.compile_model",
            bool(run_compile_model.value),
        )
        set_path(
            generated_config,
            "run.init_from",
            parse_nullable_str(run_init_from.value),
        )
        set_path(
            generated_config,
            "run.output_dir",
            parse_nullable_str(run_output_dir.value),
        )
        set_path(generated_config, "run.name", parse_nullable_str(run_name.value))
        set_path(generated_config, "logging.backend", str(logging_backend.value))
        set_path(generated_config, "logging.mode", str(logging_mode.value))
        set_path(
            generated_config,
            "logging.mirror_stdout",
            bool(logging_mirror_stdout.value),
        )
        set_path(
            generated_config,
            "logging.wandb",
            {
                "project": parse_nullable_str(logging_wandb["project"].value),
                "entity": parse_nullable_str(logging_wandb["entity"].value),
                "group": parse_nullable_str(logging_wandb["group"].value),
                "name": parse_nullable_str(logging_wandb["name"].value),
                "tags": nonempty_values(logging_wandb["tags"]),
            },
        )
        set_path(
            generated_config,
            "checkpoint.save_latest",
            bool(checkpoint_save_latest.value),
        )
        set_path(
            generated_config,
            "checkpoint.save_best",
            bool(checkpoint_save_best.value),
        )
        set_path(
            generated_config,
            "checkpoint.save_final",
            bool(checkpoint_save_final.value),
        )
        set_path(
            generated_config,
            "checkpoint.monitors",
            nonempty_values(checkpoint_monitors),
        )
    except (TypeError, ValueError) as exc:
        build_errors.append(str(exc))

    _built = build_config(generated_config, build_errors)
    generated_config = _built.raw
    experiment_config = _built.experiment
    generated_json = _built.json
    validation_error = _built.error
    return experiment_config, generated_config, generated_json, validation_error


@app.cell
def _(
    generated_json,
    load_path,
    overwrite_existing,
    save_path,
    set_action_status,
    validation_error,
):
    _current_load_path = str(load_path.value)

    def validate_config(value):
        next_value = int(value or 0) + 1
        if validation_error is None:
            set_action_status((_current_load_path, "Generated config is valid", "success"))
        else:
            set_action_status((_current_load_path, validation_error, "danger"))
        return next_value

    def save_config(value):
        next_value = int(value or 0) + 1
        if validation_error is not None:
            set_action_status(
                (
                    _current_load_path,
                    f"Not saved: config is invalid: {validation_error}",
                    "danger",
                )
            )
            return next_value
        try:
            path = save_config_file(
                str(save_path.value),
                generated_json,
                overwrite=bool(overwrite_existing.value),
            )
        except FileExistsError as exc:
            set_action_status(
                (
                    _current_load_path,
                    f"Not saved: {exc.args[0]} exists. Check overwrite to replace it.",
                    "warn",
                )
            )
            return next_value
        except OSError as exc:
            set_action_status((_current_load_path, f"Not saved: {exc}", "danger"))
            return next_value
        set_action_status((_current_load_path, f"Saved config to {path}", "success"))
        return next_value

    validate_button = mo.ui.button(value=0, on_click=validate_config, label="Validate config")
    save_button = mo.ui.button(value=0, on_click=save_config, label="Save generated config")
    return save_button, validate_button


@app.cell
def _(
    data_feedback_column_count,
    data_feedback_columns,
    data_input_column_count,
    data_input_columns,
    data_manifest_count,
    data_manifest_paths,
    data_protocol_count,
    data_protocol_mode,
    data_protocols,
    data_scaling,
    data_store_root,
    data_target_column_count,
    data_target_columns,
    get_action_status,
    load_error,
    load_path,
    loaded_schema_error,
    loader_batch_size,
    loader_cross_protocol_state_carry,
    loader_data_access,
    loader_num_workers,
    loader_prefetch_to_device,
    loader_seq_len,
    loader_stateful_n_windows,
    loader_strategy,
    overwrite_existing,
    save_button,
    save_path,
    tree_array_rows,
    validate_button,
    with_help,
):
    action_status = get_action_status()
    if load_error is not None:
        status_message, status_kind = load_error, "danger"
    elif action_status is not None and action_status[0] == str(load_path.value):
        _, status_message, status_kind = action_status
    elif loaded_schema_error is not None:
        status_message = (
            "Loaded raw JSON with schema issues; edit the fields below to repair it: "
            f"{loaded_schema_error}"
        )
        status_kind = "warn"
    else:
        status_message, status_kind = "Loaded config", "success"
    scaling_fields = {
        str(entry["column"]): {
            "input_min": entry["input_min"],
            "input_max": entry["input_max"],
            "output_min": entry["output_min"],
            "output_max": entry["output_max"],
            "transform": entry["transform"],
            "clip": entry["clip"],
        }
        for entry in data_scaling
    }
    if not scaling_fields:
        scaling_fields = {
            "status": mo.md("Select input or target columns to configure scaling.")
        }
    mo.vstack(
        [
            mo.md("# Create ML Config"),
            mo.hstack([load_path, save_path], justify="start"),
            overwrite_existing,
            mo.hstack([validate_button, save_button], justify="start", gap=0.75),
            mo.callout(status_message, kind=status_kind),
            mo.md("## data"),
            mo.tree(
                {
                    "manifest_paths": tree_array_rows(
                        data_manifest_paths,
                        data_manifest_count,
                        'Mapping from normalized manifest path to expected git commit prefix. Use JSON object entry syntax: "manifest/path.parquet": "commit-prefix".',
                    ),
                    "protocols": tree_array_rows(
                        data_protocols,
                        data_protocol_count,
                        "Dataset protocols to include. Canonical values: cycling, HPPC, RPT, EIS.",
                    ),
                    "protocol_mode": with_help(
                        data_protocol_mode,
                        "available skips protocols missing from the manifest; strict requires all requested protocols.",
                    ),
                    "store_root": with_help(
                        data_store_root,
                        "Optional data store root. Empty means null and DATA_ROOT will be used at training time.",
                    ),
                    "input_columns": tree_array_rows(
                        data_input_columns,
                        data_input_column_count,
                        "Model input columns read from normalized shards.",
                    ),
                    "target_columns": tree_array_rows(
                        data_target_columns,
                        data_target_column_count,
                        "Columns predicted by the model.",
                    ),
                    "feedback_columns": tree_array_rows(
                        data_feedback_columns,
                        data_feedback_column_count,
                        "Targets also present as inputs for autoregressive feedback.",
                    ),
                    "scaling": scaling_fields,
                },
                label="data",
            ),
            mo.md("## loader"),
            mo.tree(
                {
                    "batch_size": with_help(
                        loader_batch_size,
                        "Number of samples/windows per batch.",
                    ),
                    "seq_len": with_help(
                        loader_seq_len,
                        "Model context length before any train roll-forward expansion.",
                    ),
                    "strategy": with_help(loader_strategy, "Batch planning strategy."),
                    "stateful_n_windows": with_help(
                        loader_stateful_n_windows,
                        "-1 means whole stream; positive values chain that many consecutive windows.",
                    ),
                    "cross_protocol_state_carry": with_help(
                        loader_cross_protocol_state_carry,
                        "Whether to chain state across protocols sharing an alignment key.",
                    ),
                    "data_access": with_help(
                        loader_data_access,
                        "windowed reads windows lazily; full_in_mem preloads selected split/protocol tensors.",
                    ),
                    "num_workers": with_help(
                        loader_num_workers, "PyTorch DataLoader worker count."
                    ),
                    "prefetch_to_device": with_help(
                        loader_prefetch_to_device,
                        "If true, prefetch batches to CUDA device.",
                    ),
                },
                label="loader",
            ),
        ]
    )
    return


@app.cell
def _(
    get_layer_kind_change,
    model_bias,
    model_causal_attention,
    model_d_model,
    model_dropout,
    model_feedback_mode,
    model_head_layer_count,
    model_head_layers,
    model_input_sigma,
    model_layer_count,
    model_layers,
    model_mamba_chunk_size,
    model_mamba_d_state,
    model_mamba_expand,
    model_mamba_headdim,
    model_mamba_is_mimo,
    model_mamba_mimo_rank,
    model_mamba_ngroups,
    model_mlp_ratio,
    model_n_heads,
    model_norm,
    model_num_bins,
    model_output_parameterization,
    model_output_sigma,
    with_help,
):
    get_layer_kind_change()

    def layer_views(layers: tuple[dict[str, object], ...], count_widget: object):
        views = []
        for idx, layer in enumerate(layers):
            view = {
                "kind": mo.hstack(
                    [layer["kind"], count_widget.style({"width": "3rem"})],
                    justify="start",
                    gap=0.5,
                    align="center",
                )
                if idx == 0
                else layer["kind"],
                "residual": layer["residual"],
            }
            kind = str(layer["kind"].value)
            if kind == "reduce":
                view["mode"] = mo.md("`sum_pool`")
            elif kind in {"attention", "ffn"}:
                view["bias"] = layer["bias"]
            elif kind == "mamba":
                view.update(
                    {
                        "d_state": layer["d_state"],
                        "expand": layer["expand"],
                        "headdim": layer["headdim"],
                        "ngroups": layer["ngroups"],
                        "is_mimo": layer["is_mimo"],
                        "mimo_rank": layer["mimo_rank"],
                        "chunk_size": layer["chunk_size"],
                    }
                )
            views.append(view)
        if not views:
            views.append(
                mo.hstack(
                    [
                        mo.md("No rows"),
                        count_widget.style({"width": "3rem"}),
                    ],
                    justify="start",
                    gap=0.5,
                    align="center",
                )
            )
        return tuple(views)

    model_fields = {
        "d_model": with_help(model_d_model, "Transformer/Mamba hidden width."),
        "n_heads": with_help(
            model_n_heads, "Attention head count; d_model must be divisible by n_heads."
        ),
        "mlp_ratio": with_help(model_mlp_ratio, "Feed-forward hidden multiplier."),
        "dropout": with_help(model_dropout, "Dropout probability in [0, 1)."),
        "bias": with_help(
            model_bias,
            "Default bias for feature, attention, FFN, and output linear projections.",
        ),
        "norm": with_help(model_norm, "Normalization kind. Currently rmsnorm only."),
        "causal_attention": with_help(model_causal_attention, "Use causal attention masks."),
        "num_bins": with_help(model_num_bins, "Number of bins for binned target parameterization."),
        "input_sigma": with_help(model_input_sigma, "Input noise sigma. Must be >= 0."),
        "output_sigma": with_help(model_output_sigma, "Output noise sigma. Must be >= 0."),
        "feedback_mode": with_help(
            model_feedback_mode, "Feedback representation used during rollout."
        ),
    }
    model_fields["mamba"] = {
        "d_state": with_help(
            model_mamba_d_state, "Default Mamba d_state; ignored when no Mamba layer uses it."
        ),
        "expand": with_help(
            model_mamba_expand, "Default Mamba expand; ignored when no Mamba layer uses it."
        ),
        "headdim": with_help(
            model_mamba_headdim,
            "Default Mamba head dimension; ignored when no Mamba layer uses it.",
        ),
        "ngroups": with_help(
            model_mamba_ngroups, "Default Mamba group count; ignored when no Mamba layer uses it."
        ),
        "is_mimo": with_help(
            model_mamba_is_mimo, "Default Mamba MIMO mode; ignored when no Mamba layer uses it."
        ),
        "mimo_rank": with_help(
            model_mamba_mimo_rank, "Default Mamba MIMO rank; ignored when no Mamba layer uses it."
        ),
        "chunk_size": with_help(
            model_mamba_chunk_size, "Default Mamba chunk size; ignored when no Mamba layer uses it."
        ),
    }
    model_fields["layers"] = layer_views(model_layers, model_layer_count)
    model_fields["head_layers"] = layer_views(model_head_layers, model_head_layer_count)
    model_fields["output"] = {
        "parameterization": with_help(
            model_output_parameterization, "Output parameterization. Currently shared only."
        ),
    }

    mo.vstack(
        [
            mo.md("## model"),
            mo.tree(
                model_fields,
                label="model",
            ),
        ]
    )
    return


@app.cell
def _(
    dynamic_dict_entries,
    info_icon,
    train_epochs,
    train_grad_clip_norm,
    train_log_every_steps,
    train_log_per_epoch,
    train_masked_suffix_carry_mamba_state,
    train_masked_suffix_channel_count,
    train_masked_suffix_channels,
    train_masked_suffix_detach_between_windows,
    train_masked_suffix_enabled,
    train_masked_suffix_loss_on_masked_only,
    train_masked_suffix_roll_forward_steps,
    train_masked_suffix_suffix_steps,
    train_max_steps,
    train_validate_every_steps,
    train_validate_per_epoch,
    tree_array_rows,
    validation_group_by_count,
    validation_log_rollout_plots,
    validation_masked_suffix_carry_mamba_state,
    validation_masked_suffix_enabled,
    validation_masked_suffix_suffix_steps,
    validation_max_tf_batches,
    validation_rollout_extension_enabled,
    validation_rollout_extension_input_value_count,
    validation_rollout_extension_input_values,
    validation_rollout_extension_steps,
    validation_rollout_steps,
    validation_split_fraction,
    validation_split_group_by,
    validation_split_group_count,
    validation_split_groups,
    validation_split_strategy,
    with_help,
):
    validation_group_views = tuple(
        {
            "match": mo.hstack(
                [
                    dynamic_dict_entries(
                        group_widgets["match"],
                        group_widgets["match_count"],
                        "Columns and values that identify this validation group.",
                    ),
                    info_icon("Explicit validation groups. Empty trailing groups are ignored."),
                    validation_split_group_count.style({"width": "3rem"}),
                ],
                justify="start",
                gap=0.5,
                align="start",
            )
            if idx == 0
            else dynamic_dict_entries(
                group_widgets["match"],
                group_widgets["match_count"],
                "Columns and values that identify this validation group.",
            ),
            "rollout_start_offsets": tree_array_rows(
                tuple(group_widgets["rollout_start_offsets"]),
                group_widgets["rollout_start_offset_count"],
                "Optional rollout start offsets for this validation group.",
            ),
        }
        for idx, group_widgets in enumerate(validation_split_groups)
    )
    if not validation_group_views:
        validation_group_views = (
            mo.hstack(
                [
                    mo.md("No explicit groups"),
                    info_icon("Ignored for sample strategy; required for provide."),
                    validation_split_group_count.style({"width": "3rem"}),
                ],
                justify="start",
                gap=0.5,
            ),
        )
    validation_split_fields = {
        "strategy": with_help(validation_split_strategy, "Validation split strategy."),
        "group_by": tree_array_rows(
            validation_split_group_by,
            validation_group_by_count,
            "Manifest columns used to define train/validation groups.",
        ),
    }
    if validation_split_strategy.value != "provide":
        validation_split_fields["fraction"] = with_help(
            validation_split_fraction,
            "Sample/merge validation fraction.",
        )
    if validation_split_strategy.value != "sample":
        validation_split_fields["groups"] = validation_group_views
    rollout_extension_fields = {
        "enabled": with_help(
            validation_rollout_extension_enabled, "Append configured future input rows to rollout."
        ),
        "steps": with_help(
            validation_rollout_extension_steps,
            "Number of extension steps; ignored when disabled.",
        ),
        "input_values": dynamic_dict_entries(
            validation_rollout_extension_input_values,
            validation_rollout_extension_input_value_count,
            "Physical input values parsed as floats; ignored when disabled.",
        ),
    }
    masked_suffix_fields = {
        "enabled": with_help(train_masked_suffix_enabled, "Enable masked suffix training."),
        "channels": tree_array_rows(
            train_masked_suffix_channels,
            train_masked_suffix_channel_count,
            "Feedback channels; ignored when masked suffix is disabled.",
        ),
        "suffix_steps": with_help(
            train_masked_suffix_suffix_steps,
            "Number of suffix steps; ignored when disabled.",
        ),
        "loss_on_masked_only": with_help(
            train_masked_suffix_loss_on_masked_only,
            "Ignored when masked suffix is disabled.",
        ),
        "carry_mamba_state": with_help(
            train_masked_suffix_carry_mamba_state,
            "Ignored when masked suffix is disabled.",
        ),
        "detach_between_windows": with_help(
            train_masked_suffix_detach_between_windows,
            "Ignored when masked suffix is disabled.",
        ),
        "roll_forward_steps": with_help(
            train_masked_suffix_roll_forward_steps,
            "Ignored when masked suffix is disabled.",
        ),
    }
    mo.vstack(
        [
            mo.md("## train"),
            mo.tree(
                {
                    "epochs": with_help(
                        train_epochs,
                        "Number of training epochs unless max_steps caps training.",
                    ),
                    "max_steps": with_help(
                        train_max_steps,
                        "Optional hard training-step cap. Empty means null.",
                    ),
                    "log_per_epoch": with_help(
                        train_log_per_epoch,
                        "Logging runs per epoch when log_every_steps is empty.",
                    ),
                    "log_every_steps": with_help(
                        train_log_every_steps,
                        "Optional fixed logging cadence. Empty uses per-epoch cadence.",
                    ),
                    "validate_every_steps": with_help(
                        train_validate_every_steps,
                        "Optional fixed validation cadence. Empty uses per-epoch cadence.",
                    ),
                    "validate_per_epoch": with_help(
                        train_validate_per_epoch,
                        "Validation runs per epoch when validate_every_steps is empty.",
                    ),
                    "loss": with_help(
                        mo.md("`categorical_ce`"),
                        "Fixed current training loss capability.",
                    ),
                    "grad_clip_norm": with_help(train_grad_clip_norm, "Gradient clipping norm."),
                    "masked_suffix": masked_suffix_fields,
                },
                label="train",
            ),
            mo.md("## validation"),
            mo.tree(
                {
                    "split": {
                        **validation_split_fields,
                    },
                    "max_tf_batches": with_help(
                        validation_max_tf_batches,
                        "Maximum teacher-forced validation batches.",
                    ),
                    "rollout_steps": with_help(
                        validation_rollout_steps, "Validation rollout horizon."
                    ),
                    "log_rollout_plots": with_help(
                        validation_log_rollout_plots,
                        "Whether rollout plots are logged.",
                    ),
                    "masked_suffix": {
                        "enabled": with_help(
                            validation_masked_suffix_enabled,
                            "Validation override; inherit/null uses train setting.",
                        ),
                        "suffix_steps": with_help(
                            validation_masked_suffix_suffix_steps,
                            "Validation suffix-step override. Empty means inherit/null.",
                        ),
                        "carry_mamba_state": with_help(
                            validation_masked_suffix_carry_mamba_state,
                            "Validation state-carry override. inherit/null uses train setting.",
                        ),
                    },
                    "rollout_extension": rollout_extension_fields,
                },
                label="validation",
            ),
        ]
    )
    return


@app.cell
def _(
    checkpoint_monitor_count,
    checkpoint_monitors,
    checkpoint_save_best,
    checkpoint_save_final,
    checkpoint_save_latest,
    generated_config,
    logging_backend,
    logging_mirror_stdout,
    logging_mode,
    logging_wandb,
    optim_beta1,
    optim_beta2,
    optim_eps,
    optim_lr,
    optim_weight_decay,
    run_compile_model,
    run_device,
    run_init_from,
    run_name,
    run_output_dir,
    run_seed,
    run_use_amp,
    scheduler_kind,
    scheduler_min_lr_ratio,
    scheduler_warmup_ratio,
    tree_array_rows,
    with_help,
):
    scheduler_fields = {
        "kind": with_help(scheduler_kind, "Learning-rate scheduler kind."),
        "warmup_ratio": with_help(
            scheduler_warmup_ratio,
            "Used by linear_warmup_cosine; ignored when scheduler kind is none.",
        ),
        "min_lr_ratio": with_help(
            scheduler_min_lr_ratio,
            "Used by linear_warmup_cosine; ignored when scheduler kind is none.",
        ),
    }
    logging_fields = {
        "backend": with_help(logging_backend, "Logging backend."),
    }
    if logging_backend.value != "stdout":
        logging_fields["mirror_stdout"] = with_help(
            logging_mirror_stdout, "Mirror metrics to stdout."
        )
    if logging_backend.value == "wandb":
        logging_fields["mode"] = with_help(logging_mode, "W&B run mode.")
        logging_fields["wandb"] = {
            "project": with_help(logging_wandb["project"], "Used only by W&B; empty means null."),
            "entity": with_help(logging_wandb["entity"], "Used only by W&B; empty means null."),
            "group": with_help(logging_wandb["group"], "Used only by W&B; empty means null."),
            "name": with_help(logging_wandb["name"], "Used only by W&B; empty means null."),
            "tags": tree_array_rows(
                logging_wandb["tags"],
                logging_wandb["tags_count"],
                "Used only by the W&B backend.",
            ),
        }
    checkpoint_fields = {
        "save_latest": with_help(
            checkpoint_save_latest, "Save latest checkpoint after validation."
        ),
        "save_best": with_help(checkpoint_save_best, "Save best checkpoints by monitored metrics."),
        "save_final": with_help(checkpoint_save_final, "Save final checkpoint at training end."),
        "monitors": tree_array_rows(
            checkpoint_monitors,
            checkpoint_monitor_count,
            "Required by save_best; otherwise ignored.",
        ),
    }
    mo.vstack(
        [
            mo.md("## optim"),
            mo.tree(
                {
                    "kind": with_help(
                        mo.md("`adamw`"),
                        "Optimizer kind is fixed by this notebook.",
                    ),
                    "lr": with_help(optim_lr, "AdamW learning rate."),
                    "weight_decay": with_help(optim_weight_decay, "AdamW weight decay."),
                    "beta1": with_help(optim_beta1, "AdamW beta1."),
                    "beta2": with_help(optim_beta2, "AdamW beta2."),
                    "eps": with_help(optim_eps, "AdamW epsilon."),
                },
                label="optim",
            ),
            mo.md("## scheduler"),
            mo.tree(
                scheduler_fields,
                label="scheduler",
            ),
            mo.md("## run"),
            mo.tree(
                {
                    "device": with_help(
                        run_device,
                        "Training device string, e.g. cuda, cuda:0, cpu.",
                    ),
                    "seed": with_help(run_seed, "Random seed."),
                    "use_amp": with_help(run_use_amp, "Use CUDA automatic mixed precision."),
                    "compile_model": with_help(run_compile_model, "Wrap model in torch.compile."),
                    "init_from": with_help(
                        run_init_from,
                        "Optional checkpoint path for model initialization. Empty means null.",
                    ),
                    "output_dir": with_help(
                        run_output_dir,
                        "Run output directory. Empty means null.",
                    ),
                    "name": with_help(
                        run_name,
                        "Optional run name. Empty means generated timestamp/null.",
                    ),
                },
                label="run",
            ),
            mo.md("## logging"),
            mo.tree(
                logging_fields,
                label="logging",
            ),
            mo.md("## checkpoint"),
            mo.tree(
                checkpoint_fields,
                label="checkpoint",
            ),
            mo.md("## Raw JSON Output"),
            mo.json(generated_config, label="Generated config"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
