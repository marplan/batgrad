from __future__ import annotations

import contextlib
import copy
import io
import logging
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_PACKAGE_LOGGER_NAME = "batgrad"
_RECENT_LOG_LIMIT = 2000
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(location)s %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BLUE = "\033[34m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


@dataclass(frozen=True, slots=True)
class RecentLogRecord:
    timestamp: str
    level: str
    logger: str
    location: str
    message: str


@dataclass(slots=True)
class _LoggerState:
    level: int = logging.INFO
    configured: bool = False
    console_handler: logging.StreamHandler | None = None
    file_handler: logging.Handler | None = None
    recent_handler: logging.Handler | None = None
    recent_logs: deque[RecentLogRecord] = field(
        default_factory=lambda: deque(maxlen=_RECENT_LOG_LIMIT),
    )


_state = _LoggerState()


class LogFormatter(logging.Formatter):
    LEVELS_COLOR: ClassVar[dict[str, str]] = {
        "DEBUG": f"{_CYAN}{_BOLD}-DEBUG-{_RESET}",
        "INFO": f"{_GREEN}{_BOLD}-INFO-{_RESET}",
        "WARNING": f"{_YELLOW}{_BOLD}-WARNING-{_RESET}",
        "ERROR": f"{_RED}{_BOLD}-ERROR-{_RESET}",
        "CRITICAL": f"{_MAGENTA}{_BOLD}-CRITICAL-{_RESET}",
    }
    LEVELS_PLAIN: ClassVar[dict[str, str]] = {
        "DEBUG": "-DEBUG-",
        "INFO": "-INFO-",
        "WARNING": "-WARNING-",
        "ERROR": "-ERROR-",
        "CRITICAL": "-CRITICAL-",
    }

    def __init__(
        self,
        fmt: str = _DEFAULT_FORMAT,
        datefmt: str = _DEFAULT_DATE_FORMAT,
        *,
        use_colors: bool = False,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors
        self._levels = self.LEVELS_COLOR if use_colors else self.LEVELS_PLAIN

    @override
    def format(self, record: logging.LogRecord) -> str:
        copied = copy.copy(record)
        plain_name = copied.name
        copied.levelname = self._levels.get(copied.levelname, copied.levelname)
        if self.use_colors:
            copied.name = f"{_BLUE}{plain_name}{_RESET}"
            copied.location = f"{copied.name}{_DIM}:{copied.lineno}:{_RESET}"
        else:
            copied.location = f"{plain_name}:{copied.lineno}:"
        return super().format(copied)

    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC).astimezone()
        rendered = dt.strftime(datefmt) if datefmt is not None else dt.isoformat(timespec="seconds")
        if self.use_colors:
            return f"{_DIM}{rendered}{_RESET}"
        return rendered


class _RecentLogHandler(logging.Handler):
    @override
    def emit(self, record: logging.LogRecord) -> None:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).astimezone().strftime("%H:%M:%S")
        _state.recent_logs.append(
            RecentLogRecord(
                timestamp=timestamp,
                level=record.levelname,
                logger=record.name,
                location=f"{record.module}:{record.lineno}",
                message=record.getMessage(),
            ),
        )


class ListHandler(logging.Handler):
    def __init__(
        self,
        level: int = logging.NOTSET,
        record_callback: Callable[[logging.LogRecord], None] | None = None,
    ) -> None:
        super().__init__(level=level)
        self.records: list[logging.LogRecord] = []
        self.record_callback = record_callback

    @override
    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if self.record_callback is not None:
            self.record_callback(record)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logger(
    level: str | int = logging.INFO,
    *,
    console: bool = True,
    file_path: str | Path | None = None,
    file_level: str | int | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    level_int = _log_level_to_int(level)
    _state.level = level_int

    package_logger = _package_logger()
    package_logger.setLevel(min(level_int, logging.INFO) if file_path is not None else level_int)
    package_logger.propagate = False

    _ensure_recent_handler(package_logger, level_int)
    _configure_console_handler(package_logger, level_int, console=console)
    if file_path is not None:
        enable_process_file_logging(
            file_path,
            level=file_level,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )
    elif _state.file_handler is not None:
        _state.file_handler.setLevel(level_int)
    _state.configured = True


def set_level(level: str | int) -> None:
    level_int = _log_level_to_int(level)
    _state.level = level_int
    package_logger = _package_logger()
    if _state.file_handler is not None:
        package_logger.setLevel(min(level_int, logging.INFO))
    else:
        package_logger.setLevel(level_int)
    for handler in package_logger.handlers:
        if handler is _state.file_handler:
            handler.setLevel(min(level_int, logging.INFO))
        else:
            handler.setLevel(level_int)


def enable_process_file_logging(
    path: str | Path,
    level: str | int | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> Path:
    resolved_path = Path(path).resolve()
    package_logger = _package_logger()
    package_logger.setLevel(min(_state.level, logging.INFO))
    package_logger.propagate = False

    if _state.file_handler is not None:
        package_logger.removeHandler(_state.file_handler)
        _state.file_handler.close()

    handler_level = min(_state.level, logging.INFO) if level is None else _log_level_to_int(level)
    handler = _build_file_handler(
        path=resolved_path,
        level=handler_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    package_logger.addHandler(handler)
    _state.file_handler = handler
    return resolved_path


def get_recent_logs(limit: int = 200) -> list[RecentLogRecord]:
    bounded = max(1, int(limit))
    return list(_state.recent_logs)[-bounded:]


def clear_recent_logs() -> None:
    _state.recent_logs.clear()


@contextmanager
def capture_output(
    log_level: str | int = logging.DEBUG,
    *,
    record_callback: Callable[[logging.LogRecord], None] | None = None,
) -> Iterator[tuple[io.StringIO, io.StringIO, list[logging.LogRecord]]]:
    package_logger = _package_logger()
    out = io.StringIO()
    err = io.StringIO()
    handler = ListHandler(_log_level_to_int(log_level), record_callback)

    old_handlers = package_logger.handlers[:]
    old_propagate = package_logger.propagate
    old_level = package_logger.level
    package_logger.handlers = [handler]
    package_logger.propagate = False
    package_logger.setLevel(handler.level)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield out, err, handler.records
    finally:
        package_logger.handlers = old_handlers
        package_logger.propagate = old_propagate
        package_logger.setLevel(old_level)


@contextmanager
def suppress_warnings_logs(name: str) -> Iterator[None]:
    logger = get_logger(name)
    original_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def _package_logger() -> logging.Logger:
    return logging.getLogger(_PACKAGE_LOGGER_NAME)


def _configure_console_handler(
    package_logger: logging.Logger,
    level: int,
    *,
    console: bool,
) -> None:
    if not console:
        if _state.console_handler is not None:
            package_logger.removeHandler(_state.console_handler)
            _state.console_handler.close()
            _state.console_handler = None
        return

    if _state.console_handler is None:
        _state.console_handler = _build_stream_handler(level)
        package_logger.addHandler(_state.console_handler)
    else:
        if _state.console_handler not in package_logger.handlers:
            package_logger.addHandler(_state.console_handler)
        _state.console_handler.setLevel(level)


def _ensure_recent_handler(package_logger: logging.Logger, level: int) -> None:
    if _state.recent_handler is None:
        _state.recent_handler = _RecentLogHandler(level=level)
        package_logger.addHandler(_state.recent_handler)
    else:
        if _state.recent_handler not in package_logger.handlers:
            package_logger.addHandler(_state.recent_handler)
        _state.recent_handler.setLevel(level)


def _build_stream_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(LogFormatter(use_colors=_is_tty()))
    handler.setLevel(level)
    return handler


def _build_file_handler(
    path: Path,
    level: int,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(LogFormatter(use_colors=False))
    handler.setLevel(level)
    return handler


def _log_level_to_int(level: str | int) -> int:
    if isinstance(level, str):
        normalized = level.upper()
        if normalized == "WARN":
            normalized = "WARNING"
        value = getattr(logging, normalized, None)
        if isinstance(value, int):
            return value
        raise ValueError(f"Unrecognized log level {level!r}")
    if level in {
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    }:
        return level
    raise ValueError(f"Unrecognized log level {level!r}")


def _is_tty() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
