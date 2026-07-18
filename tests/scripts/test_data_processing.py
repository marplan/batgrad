from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from batgrad.data.datasets.registry import dataset_ids
from scripts import data_processing


def test_processing_parser_defaults_and_all_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/data/configured")
    parser = data_processing._parser()
    args = parser.parse_args(["--ingest", "all"])

    assert args.store == Path("/data/configured")
    assert args.scratch_store == Path(tempfile.gettempdir())
    assert args.n_jobs == -1
    assert data_processing._expand_selectors(args.ingest, parser, "--ingest") == dataset_ids()


def test_all_cannot_be_combined_with_explicit_dataset() -> None:
    parser = data_processing._parser()

    with pytest.raises(SystemExit):
        data_processing._expand_selectors(
            ["all", dataset_ids()[0]],
            parser,
            "--ingest",
        )


def test_main_runs_all_ingest_stages_before_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str]] = []

    class FakeDataset:
        def __init__(self, dataset_id: str) -> None:
            self.dataset_id = dataset_id

        def ingest(self, *_args, **_kwargs) -> None:
            events.append(("ingest", self.dataset_id))

        def normalize(self, *_args, **_kwargs) -> None:
            events.append(("normalize", self.dataset_id))

    monkeypatch.setattr(data_processing, "configure_logging", lambda: None)
    monkeypatch.setattr(data_processing, "_require_ingest_tasks", lambda *_args: ())
    monkeypatch.setattr(data_processing, "_require_ingested_manifest", lambda *_args: None)
    monkeypatch.setattr(
        data_processing,
        "LocalDataProcessingStore",
        lambda path, create=False: (path, create),
    )
    monkeypatch.setattr(data_processing, "get_dataset", FakeDataset)
    monkeypatch.setattr(
        "sys.argv",
        ["data_processing.py", "--ingest", "all", "--normalize", "all"],
    )

    data_processing.main()

    assert events == [
        *(("ingest", dataset_id) for dataset_id in dataset_ids()),
        *(("normalize", dataset_id) for dataset_id in dataset_ids()),
    ]
