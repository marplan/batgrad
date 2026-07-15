from __future__ import annotations

import logging
from contextlib import contextmanager
from html import escape
from typing import TYPE_CHECKING

import marimo as mo

from batgrad.logging import LogFormatter, capture_output

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


@contextmanager
def capture_log_lines(
    destination: list[str],
    progress_callback: Callable[[str], None] | None = None,
) -> Iterator[None]:
    formatter = LogFormatter(use_colors=False)

    def capture_record(record: logging.LogRecord) -> None:
        line = formatter.format(record)
        destination.append(line)
        if progress_callback is not None:
            progress_callback(line)
        del destination[:-2000]

    with capture_output(
        logging.DEBUG,
        record_callback=capture_record,
    ) as (stdout, stderr, _records):
        try:
            yield
        finally:
            output_lines = [
                line
                for line in (*stdout.getvalue().splitlines(), *stderr.getvalue().splitlines())
                if line
            ]
            destination.extend(output_lines)
            if progress_callback is not None:
                for line in output_lines:
                    progress_callback(line)
            del destination[:-2000]


def log_view(
    lines: Sequence[str],
    *,
    title: str,
    empty: str,
) -> mo.Html:
    text = "\n".join(lines) if lines else empty
    return mo.Html(
        '<div style="font-size: 0.875rem;">'
        f'<div style="font-weight: 600; margin-bottom: 0.25rem;">{escape(title)}</div>'
        '<div style="max-height: 420px; min-height: 180px; overflow-y: auto; '
        "display: flex; flex-direction: column-reverse; border: 1px solid var(--slate-6); "
        'border-radius: 4px; background: var(--slate-1);">'
        '<pre style="flex: 0 0 auto; margin: 0; padding: 0.75rem; white-space: pre-wrap; '
        f'overflow-wrap: anywhere;">{escape(text)}</pre></div></div>'
    )
