from __future__ import annotations

import math

import polars as pl
import pytest

from batgrad.contracts.mapping import BaseColumns
from batgrad.data.transforms.checks import (
    ColumnBoundsCheckSpec,
    DomainAxisCheckSpec,
    ImpedanceComponentsCheckSpec,
    MissingCheckSpec,
    TimeCheckSpec,
    apply_checks_bounded_chunk,
    apply_checks_full_task,
)


def test_missing_and_bounds_checks_annotate_rows() -> None:
    frame = pl.DataFrame({str(BaseColumns.curr): [1.0, None], str(BaseColumns.volt): [6.0, 3.0]})
    checked, violations = apply_checks_full_task(
        frame.lazy(),
        (),
        (
            MissingCheckSpec((BaseColumns.curr,)),
            ColumnBoundsCheckSpec({BaseColumns.volt: (2.0, 5.0)}),
        ),
        annotate=True,
    )
    result = checked.collect()
    assert violations == ()
    assert result[str(BaseColumns.ann_reasons)].to_list() == ["above column maximum", "missing"]


def test_checks_report_violations_when_not_annotating() -> None:
    frame = pl.DataFrame({str(BaseColumns.curr): pl.Series([None], dtype=pl.Float64)})
    _checked, violations = apply_checks_full_task(
        frame.lazy(), (), (MissingCheckSpec((BaseColumns.curr,)),), annotate=False
    )
    assert violations == ((str(BaseColumns.curr), "missing"),)


def test_time_check_rebuilds_time_and_flags_duplicate_and_big_dt() -> None:
    frame = pl.DataFrame({str(BaseColumns.time): [0.0, 1.0, 1.0, 10.0]})
    checked, violations = TimeCheckSpec(BaseColumns.time, BaseColumns.dt, max_dt_s=5.0).apply_full(
        frame.lazy(), (), annotate=True
    )
    result = checked.collect()
    assert violations == ()
    assert result[str(BaseColumns.time)].to_list() == [0.0, 1.0]
    assert result[str(BaseColumns.ann_reasons)].to_list() == [
        None,
        "duplicate time steps\x1fbig time diff",
    ]
    with pytest.raises(ValueError, match="No time column"):
        TimeCheckSpec(BaseColumns.time, BaseColumns.dt).apply_full(
            pl.DataFrame({"x": [1]}).lazy(),
            (),
        )


def test_time_check_bounded_carries_state_across_chunks() -> None:
    check = TimeCheckSpec(BaseColumns.time, BaseColumns.dt, max_dt_s=5.0)
    state = check.init_state()
    first, _ = check.apply_chunk(pl.DataFrame({str(BaseColumns.time): [0.0]}), state)
    second, _ = check.apply_chunk(pl.DataFrame({str(BaseColumns.time): [1.0, 2.0]}), state)
    assert first.height == 0
    assert second[str(BaseColumns.time)].to_list() == [0.0, 1.0]


def test_domain_axis_check_flags_non_increasing_values() -> None:
    check = DomainAxisCheckSpec(BaseColumns.freq, zero_replacement=0.1, enforce_positive=True)
    frame = pl.DataFrame({str(BaseColumns.freq): [0.0, 1.0, 1.0, -1.0]})
    result, _ = check.apply_full(frame.lazy(), (), annotate=True)
    reasons = result.collect()[str(BaseColumns.ann_reasons)].to_list()
    assert reasons == [None, None, "invalid domain x-axis", "invalid domain x-axis"]

    state = check.init_state()
    chunk, _ = apply_checks_bounded_chunk(frame, (check,), (state,), annotate=True)
    assert chunk[str(BaseColumns.ann_reasons)].to_list() == reasons


def test_impedance_components_are_derived_from_rectangular_and_polar() -> None:
    rectangular = pl.DataFrame({str(BaseColumns.z_real): [3.0], str(BaseColumns.z_imag): [4.0]})
    result, _ = ImpedanceComponentsCheckSpec().apply_full(rectangular.lazy(), ())
    row = result.collect().row(0, named=True)
    assert row[str(BaseColumns.z_mag)] == 5.0
    assert math.isclose(row[str(BaseColumns.z_phase)], 53.13010235415598)

    polar = pl.DataFrame({str(BaseColumns.z_mag): [1.0], str(BaseColumns.z_phase): [90.0]})
    result, _ = ImpedanceComponentsCheckSpec().apply_full(polar.lazy(), ())
    assert math.isclose(result.collect()[str(BaseColumns.z_imag)][0], 1.0)
    with pytest.raises(ValueError, match="EIS data requires"):
        ImpedanceComponentsCheckSpec().apply_full(pl.DataFrame({"x": [1]}).lazy(), ())
