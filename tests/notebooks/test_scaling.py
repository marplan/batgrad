from __future__ import annotations

import marimo as mo

from batgrad.contracts.mapping import BaseColumns
from notebooks._support.config_helpers import scaling_drafts
from notebooks._support.dataloader_helpers import (
    BATCH_STRATEGY_OPTIONS,
    default_input_columns,
    default_target_columns,
)


def test_scaling_drafts_prefer_loaded_values_and_fill_unknown_columns() -> None:
    drafts = scaling_drafts(
        (BaseColumns.volt, "custom", BaseColumns.volt),
        (
            {
                "column": BaseColumns.volt,
                "input_min": 2.0,
                "input_max": 5.0,
                "clip": True,
            },
        ),
    )

    assert tuple(rule["column"] for rule in drafts) == (BaseColumns.volt, "custom")
    assert drafts[0]["input_min"] == 2.0
    assert drafts[0]["clip"] is True
    assert drafts[0]["output_min"] == -1.0
    assert drafts[1]["input_min"] == ""
    assert drafts[1]["output_min"] == -1.0


def test_dataloader_defaults_match_baseline_config() -> None:
    columns = tuple(
        str(column)
        for column in (
            BaseColumns.time,
            BaseColumns.dt,
            BaseColumns.curr,
            BaseColumns.crate,
            BaseColumns.volt,
            BaseColumns.temp,
            BaseColumns.amb_temp,
            BaseColumns.a_heat,
        )
    )

    assert default_input_columns(columns) == tuple(
        str(column)
        for column in (
            BaseColumns.dt,
            BaseColumns.crate,
            BaseColumns.volt,
            BaseColumns.temp,
            BaseColumns.amb_temp,
            BaseColumns.a_heat,
        )
    )
    assert default_target_columns(columns) == (
        str(BaseColumns.volt),
        str(BaseColumns.temp),
    )


def test_dataloader_strategy_labels_map_to_loader_values() -> None:
    assert BATCH_STRATEGY_OPTIONS == {
        "Shuffled protocol groups": "shuffled_protocol_groups",
        "Sequential debug": "sequential",
    }
    strategy = mo.ui.dropdown(
        options=BATCH_STRATEGY_OPTIONS,
        value="Shuffled protocol groups",
    )
    assert strategy.value == "shuffled_protocol_groups"
