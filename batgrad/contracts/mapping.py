from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, TypeVar, cast

import polars as pl

ValuesT = TypeVar("ValuesT")
ColumnDType = type[pl.DataType] | pl.DataType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class MissingValues:
    """Sentinel for mapping specs without declared fixed values."""

    __slots__ = ()


MISSING_VALUES = MissingValues()


class DatasetStageId(StrEnum):
    """Pipeline stage identifiers used in paths, specs, and manifests."""

    raw = "raw"
    ingested = "ingested"
    normalized = "normalized"


class DatasetTypeId(StrEnum):
    """Dataset origin labels used by dataset registry metadata."""

    published = "published"
    synthetic = "synthetic"


class DatasetProtocolId(StrEnum):
    """Canonical battery protocol labels used across processing stages."""

    cycling = "cycling"
    hppc = "HPPC"
    rpt = "RPT"
    eis = "EIS"


class GitValues(NamedTuple):
    """Fixed git state values written to metadata."""

    clean: Literal["clean"] = "clean"
    dirty: Literal["dirty"] = "dirty"
    na: Literal["na"] = "na"


class CheckReasonValues(NamedTuple):
    """Shared validation annotation reasons."""

    missing: Literal["missing"] = "missing"
    col_min: Literal["below column minimum"] = "below column minimum"
    col_max: Literal["above column maximum"] = "above column maximum"
    dup_time: Literal["duplicate time steps"] = "duplicate time steps"
    big_dt: Literal["big time diff"] = "big time diff"
    invalid_axis: Literal["invalid domain x-axis"] = "invalid domain x-axis"


class SplitValues(NamedTuple):
    """Shared split labels for train and validation partitions."""

    train: Literal["train"] = "train"
    val: Literal["val"] = "val"


class MappingSpec[ValuesT](str):
    """Canonical column name with dtype, aliases, parser, and optional values.

    `MappingSpec` is a `str` subclass, so it can be passed directly to Polars
    selectors and aliases while carrying schema metadata for validation and raw
    column discovery. Common columns live in `BaseColumns`; dataset mapping
    modules derive dataset-specific variants with aliases or parsers.

    Examples:
        >>> curr = MappingSpec("Current [A]", dtype=pl.Float64)
        >>> str(curr)
        'Current [A]'
        >>> curr.dtype
        Float64

        >>> raw_time = BaseColumns.time.with_alias("Test_Time(s)", "Test Time (s)")
        >>> raw_time.matching_name(["Test Time (s)"])
        'Test Time (s)'

        >>> parsed_time = raw_time.with_parser(lambda col: pl.col(col).cast(pl.Float64))
        >>> parsed_time.parser is not None
        True

        >>> git_status = BaseColumns.git_status
        >>> git_status.values.clean
        'clean'
    """

    __slots__ = (
        "_values",
        "alias",
        "description",
        "dtype",
        "parser",
    )

    alias: tuple[str, ...]
    dtype: pl.DataType
    description: str | None
    parser: Callable[[str], pl.Expr] | None
    _values: ValuesT | MissingValues

    def __new__(
        cls,
        name: str,
        *,
        dtype: ColumnDType,
        alias: str | Sequence[str] | None = None,
        description: str | None = None,
        parser: Callable[[str], pl.Expr] | None = None,
        values: ValuesT | MissingValues = MISSING_VALUES,
    ) -> Self:
        """Create a string column spec and attach schema metadata.

        Args:
            name: Canonical column name used in processed outputs.
            dtype: Polars dtype stored for validation and schema construction.
            alias: Additional raw/source column names that may map to `name`.
            description: Optional human-readable column description.
            parser: Optional expression factory for parsing a raw column.
            values: Optional namespace of fixed values for this column.
        """
        instance = cast("Self", super().__new__(cls, name))

        if alias is None:
            aliases = ()
        elif isinstance(alias, str):
            aliases = (alias,)
        else:
            aliases = tuple(alias)

        instance.alias = (name, *(a for a in aliases if a != name))
        instance.dtype = cls._normalize_dtype(dtype)
        instance.description = description
        instance.parser = parser
        instance._values = values
        return instance

    def __getnewargs_ex__(self) -> tuple[tuple[str], dict[str, object]]:
        """Preserve mapping metadata when specs are pickled for workers.

        Returns:
            Positional and keyword constructor arguments used to recreate the
            spec during unpickling.
        """
        return (
            (str(self),),
            {
                "dtype": self.dtype,
                "alias": self.alias[1:],
                "description": self.description,
                "parser": self.parser,
                "values": self._values,
            },
        )

    @classmethod
    def _normalize_dtype(cls, dtype: ColumnDType) -> pl.DataType:
        """Accept scalar dtype classes and nested dtype instances; store instances only."""
        if isinstance(dtype, type):
            return dtype()
        return dtype

    def with_alias(self, *alias: str) -> MappingSpec[ValuesT]:
        """Return a copy with dataset-specific source column aliases.

        Args:
            alias: Raw/source column names that should match this canonical spec.

        Returns:
            A spec with the same canonical name and metadata, but the supplied
            source aliases.
        """
        return MappingSpec[ValuesT](
            str(self),
            dtype=self.dtype,
            alias=alias,
            description=self.description,
            parser=self.parser,
            values=self._values,
        )

    def with_parser(self, parser: Callable[[str], pl.Expr]) -> MappingSpec[ValuesT]:
        """Return a copy with a raw-column parser expression factory.

        Args:
            parser: Callable receiving the matched raw column name and returning
                a Polars expression for the parsed canonical column.

        Returns:
            A spec with the same canonical name, aliases, and values, but the
            supplied parser.
        """
        return MappingSpec[ValuesT](
            str(self),
            dtype=self.dtype,
            alias=self.alias[1:],
            description=self.description,
            parser=parser,
            values=self._values,
        )

    def with_values[NewValuesT](self, values: NewValuesT) -> MappingSpec[NewValuesT]:
        """Return a copy with fixed named values attached.

        Args:
            values: Namespace of allowed or conventional values for the column.

        Returns:
            A spec with the same canonical name, aliases, parser, and dtype, but
            the supplied values namespace.
        """
        return MappingSpec[NewValuesT](
            str(self),
            dtype=self.dtype,
            alias=self.alias[1:],
            description=self.description,
            parser=self.parser,
            values=values,
        )

    @property
    def values(self) -> ValuesT:
        """Fixed named values declared for this column.

        Returns:
            The values namespace attached to this spec.

        Raises:
            TypeError: If no values namespace was declared.
        """
        if isinstance(self._values, MissingValues):
            raise TypeError(f"Column {self!r} has no values")
        return self._values

    def matching_name(self, columns: set[str] | tuple[str, ...] | list[str]) -> str | None:
        """Return the first available column matching this spec or its aliases.

        Args:
            columns: Available column names to search. Matching is
                case-insensitive and checks the canonical name first, followed by
                aliases in declaration order.

        Returns:
            The original available column name when a match is found, otherwise
            `None`.
        """
        available = {column.casefold(): column for column in columns}
        for alias in self.alias:
            match = available.get(alias.casefold())
            if match is not None:
                return match
        return None

    def has_match(self, columns: str | set[str] | tuple[str, ...] | list[str]) -> bool:
        """Return whether any available column matches this spec or its aliases.

        Args:
            columns: One column name or a collection of available column names.

        Returns:
            `True` when `matching_name` would return a column name.
        """
        if isinstance(columns, str):
            columns = (columns,)
        return self.matching_name(columns) is not None


class BaseColumns:
    """Namespace of canonical column specs reused across datasets and stages.

    Add columns here when they are shared by processing stages, metadata layouts,
    transforms, or multiple datasets. Dataset-specific raw names should usually be
    attached with `MappingSpec.with_alias` in the dataset mapping module.
    """

    set_id = MappingSpec("dataset id", dtype=pl.String, description="Source dataset identifier.")
    cell_id = MappingSpec(
        "cell id", dtype=pl.String, description="Cell identifier within a dataset."
    )
    cidx = MappingSpec(
        "cycle index",
        dtype=pl.Int64,
        description="Logical cycle/test grouping key.",
    )
    ccidx = MappingSpec(
        "Cycle index",
        dtype=pl.Int64,
        description="Cell tester raw data cycle index.",
    )
    path = MappingSpec("file path", dtype=pl.String, description="Path to a generated data file.")
    row_n = MappingSpec(
        "row count",
        dtype=pl.Int64,
        description="Number of rows in a file or segment.",
    )
    split: MappingSpec[SplitValues] = MappingSpec(
        "split",
        dtype=pl.String,
        description="Dataset split label.",
        values=SplitValues(),
    )

    anns = MappingSpec(
        "annotations",
        dtype=pl.List(pl.Struct({"column": pl.String, "reason": pl.String})),
        description="Validation annotations grouped by column and reason.",
    )
    ann_cols = MappingSpec(
        "annotation columns",
        dtype=pl.String,
        description="Annotated column names.",
    )
    ann_reasons: MappingSpec[CheckReasonValues] = MappingSpec(
        "annotation reasons",
        dtype=pl.String,
        description="Reasons attached to validation annotations.",
        values=CheckReasonValues(),
    )

    dev = MappingSpec("Device ID", dtype=pl.String, description="Cycler device identifier.")
    test = MappingSpec("Test ID", dtype=pl.Int64, description="Cycler test identifier.")
    chan = MappingSpec("Channel ID", dtype=pl.Int64, description="Cycler channel identifier.")
    step = MappingSpec("Step index", dtype=pl.Int64, description="Cycler step index.")
    step_t = MappingSpec(
        "Step time [s]",
        dtype=pl.Float64,
        description="Elapsed time within a step.",
    )
    step_id = MappingSpec("Step ID", dtype=pl.Int64, description="Cycler step identifier.")
    cyc = MappingSpec("Cycle ID", dtype=pl.Int64, description="Raw cycler cycle identifier.")
    pt = MappingSpec("Pt", dtype=pl.Int64)

    cap_nom: MappingSpec[float | int | None] = MappingSpec(
        "Nominal capacity [Ah]",
        dtype=pl.Float64,
        description="Nominal cell capacity.",
    )
    raw_paths = MappingSpec(
        "raw file paths",
        dtype=pl.List(pl.String),
        description="Raw files for a task.",
    )
    parq_segs = MappingSpec(
        "parquet segments",
        dtype=pl.List(
            pl.Struct(
                {
                    "file path": pl.String,
                    "row start": pl.Int64,
                    "row count": pl.Int64,
                },
            ),
        ),
        description="Parquet file segments backing a task.",
    )
    norm_segs = MappingSpec(
        "normalized segments",
        dtype=pl.List(
            pl.Struct(
                {
                    "file path": pl.String,
                    "row start": pl.Int64,
                    "row count": pl.Int64,
                },
            ),
        ),
        description="Normalized file segments backing a task.",
    )
    proto = MappingSpec("protocol", dtype=pl.String, description="Battery protocol label.")
    row0 = MappingSpec(
        "row start",
        dtype=pl.Int64,
        description="Segment start row in the source stream.",
    )
    order = MappingSpec(
        "ingest order",
        dtype=pl.Int64,
        description="Stable raw ingest ordering key.",
    )
    soc_pct = MappingSpec("SOC [%]", dtype=pl.Float64)
    resamp = MappingSpec(
        "resampling method",
        dtype=pl.String,
        description="Resampling algorithm name.",
    )
    resamp_args = MappingSpec(
        "resampling params",
        dtype=pl.String,
        description="Serialized resampling parameters.",
    )
    time_conv = MappingSpec(
        "time convention",
        dtype=pl.String,
        description="Time-axis convention used for output.",
    )
    stage = MappingSpec(
        "processing stage",
        dtype=pl.String,
        description="Pipeline stage that produced the data.",
    )
    git_commit = MappingSpec(
        "git commit",
        dtype=pl.String,
        description="Git commit hash for generated data, or na when unavailable.",
    )
    git_status: MappingSpec[GitValues] = MappingSpec(
        "git status",
        dtype=pl.String,
        description="Git cleanliness state for generated data.",
        values=GitValues(),
    )
    manifest = MappingSpec(
        "manifest path",
        dtype=pl.String,
        description="Manifest file path for the output.",
    )
    protos = MappingSpec(
        "protocols",
        dtype=pl.List(pl.String),
        description="Protocols present in a dataset/stage.",
    )

    time = MappingSpec("Time [s]", dtype=pl.Float64, description="Protocol elapsed time.")
    dt = MappingSpec(
        "Time diff [s]",
        dtype=pl.Float64,
        description="Difference to the previous time step.",
    )
    dtime = MappingSpec("Date time", dtype=pl.Datetime, description="Measurement timestamp.")
    curr = MappingSpec("Current [A]", dtype=pl.Float64, description="Cell current.")
    crate = MappingSpec(
        "Current [C-rate]",
        dtype=pl.Float64,
        description="Current normalized by nominal capacity.",
    )
    volt = MappingSpec(
        "Terminal voltage [V]",
        dtype=pl.Float64,
        description="Cell terminal voltage.",
    )
    temp_1 = MappingSpec(
        "Auxiliary temperature 1 [degC]",
        dtype=pl.Float64,
        description="First auxiliary temperature channel.",
    )
    temp_2 = MappingSpec(
        "Auxiliary temperature 2 [degC]",
        dtype=pl.Float64,
        description="Second auxiliary temperature channel.",
    )
    temp_3 = MappingSpec(
        "Auxiliary temperature 3 [degC]",
        dtype=pl.Float64,
        description="Third auxiliary temperature channel.",
    )
    temp = MappingSpec(
        "Surface temperature [degC]",
        dtype=pl.Float64,
        description="Auxiliary temperature channel.",
    )
    amb_temp = MappingSpec(
        "Ambient temperature [degC]",
        dtype=pl.Float64,
        description="Ambient test temperature.",
    )
    a_heat = MappingSpec(
        "Cooling alpha [W.m-2.K-1]",
        dtype=pl.Float64,
        description="Heat transfer coefficient.",
    )

    freq = MappingSpec("Frequency [Hz]", dtype=pl.Float64, description="EIS excitation frequency.")
    z_mag = MappingSpec(
        "Impedance magnitude [Ohm]",
        dtype=pl.Float64,
        description="Impedance magnitude.",
    )
    z_phase = MappingSpec(
        "Impedance phase [deg]",
        dtype=pl.Float64,
        description="Impedance phase angle.",
    )
    z_real = MappingSpec(
        "Impedance real [Ohm]",
        dtype=pl.Float64,
        description="Real impedance component.",
    )
    z_imag = MappingSpec(
        "Impedance imaginary [Ohm]",
        dtype=pl.Float64,
        description="Imaginary impedance component.",
    )

    cap_chg = MappingSpec(
        "Charge capacity [Ah]",
        dtype=pl.Float64,
        description="Integrated charge capacity.",
    )
    cap_dchg = MappingSpec(
        "Discharge capacity [Ah]",
        dtype=pl.Float64,
        description="Integrated discharge capacity.",
    )
    eng_chg = MappingSpec(
        "Charge energy [Wh]",
        dtype=pl.Float64,
        description="Integrated charge energy.",
    )
    eng_dchg = MappingSpec(
        "Discharge energy [Wh]",
        dtype=pl.Float64,
        description="Integrated discharge energy.",
    )
    res = MappingSpec(
        "Internal resistance [Ohm]",
        dtype=pl.Float64,
        description="Internal resistance estimate.",
    )
    acr = MappingSpec("ACR [Ohm]", dtype=pl.Float64)
    dvdt = MappingSpec(
        "dV/dt [V.s-1]",
        dtype=pl.Float64,
        description="Voltage derivative over time.",
    )
