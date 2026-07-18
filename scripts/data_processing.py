from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from batgrad.contracts.mapping import DatasetStageId
from batgrad.data.datasets.registry import DatasetId, dataset_ids, get_dataset
from batgrad.data.processing.manifests import load_stage_manifest
from batgrad.data.processing.normalize import NormalizeStageConfig
from batgrad.data.processing.raw import IngestStageConfig, IngestStageSpec, IngestTask
from batgrad.logging import configure_logging
from batgrad.storage.local import LocalDataProcessingStore

if TYPE_CHECKING:
    from batgrad.data.datasets.config import Dataset

_ALL_DATASETS = "all"
type DatasetSelector = DatasetId | Literal["all"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batgrad data processing stages")
    selectors = (*dataset_ids(), _ALL_DATASETS)
    parser.add_argument(
        "--ingest",
        choices=selectors,
        metavar="DATASET_ID",
        nargs="+",
        help="dataset IDs to ingest, or 'all'",
    )
    parser.add_argument(
        "--normalize",
        choices=selectors,
        metavar="DATASET_ID",
        nargs="+",
        help="dataset IDs to normalize, or 'all'",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(os.getenv("DATA_ROOT") or "/data"),
        help="input and output data store (default: DATA_ROOT or /data)",
    )
    parser.add_argument(
        "--scratch-store",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="scratch data store (default: system temporary directory)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="parallel jobs per stage (default: -1)"
    )
    return parser


def _expand_selectors(
    selectors: list[DatasetSelector] | None,
    parser: argparse.ArgumentParser,
    option: str,
) -> tuple[DatasetId, ...]:
    if selectors is None:
        return ()
    if _ALL_DATASETS in selectors:
        if len(selectors) != 1:
            parser.error(f"{option}: 'all' cannot be combined with dataset IDs")
        return dataset_ids()
    return tuple(dict.fromkeys(cast("list[DatasetId]", selectors)))


def _require_ingest_tasks(
    dataset: Dataset,
    store: LocalDataProcessingStore,
    parser: argparse.ArgumentParser,
) -> tuple[IngestTask, ...]:
    adapter = dataset.raw_adapter
    raw_spec = dataset.spec.processing_stages.get(DatasetStageId.ingested)
    raw_root = dataset.spec.source_root(DatasetStageId.raw)
    if adapter is None or not isinstance(raw_spec, IngestStageSpec):
        parser.error(f"dataset {dataset.spec.dataset_id!r} does not support ingestion")
    try:
        tasks = adapter.plan_raw_tasks(store, raw_spec)
    except FileNotFoundError:
        parser.error(
            f"cannot ingest {dataset.spec.dataset_id!r}: raw data directory is missing: "
            f"{store.resolve(raw_root)}"
        )
    if not tasks:
        parser.error(
            f"cannot ingest {dataset.spec.dataset_id!r}: no matching raw files found under "
            f"{store.resolve(raw_root)}"
        )
    return tasks


def _require_ingested_manifest(
    dataset: Dataset,
    store: LocalDataProcessingStore,
    parser: argparse.ArgumentParser,
) -> None:
    manifest_path = dataset.spec.manifest(DatasetStageId.ingested)
    resolved = Path(store.resolve(manifest_path))
    if not resolved.is_file():
        parser.error(
            f"cannot normalize {dataset.spec.dataset_id!r}: ingested manifest is missing: "
            f"{resolved}"
        )
    if load_stage_manifest(dataset.spec, store, DatasetStageId.ingested).is_empty():
        parser.error(
            f"cannot normalize {dataset.spec.dataset_id!r}: ingested manifest is empty: {resolved}"
        )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    ingest_ids = _expand_selectors(args.ingest, parser, "--ingest")
    normalize_ids = _expand_selectors(args.normalize, parser, "--normalize")
    if not ingest_ids and not normalize_ids:
        parser.error("at least one of --ingest or --normalize is required")

    configure_logging()
    store = LocalDataProcessingStore(args.store)
    scratch_store = LocalDataProcessingStore(args.scratch_store, create=True)
    ingest_config = IngestStageConfig(n_jobs=args.n_jobs)
    normalize_config = NormalizeStageConfig(n_jobs=args.n_jobs)

    for dataset_id in ingest_ids:
        dataset = get_dataset(dataset_id)
        tasks = _require_ingest_tasks(dataset, store, parser)
        dataset.ingest(
            store,
            store,
            ingest_config,
            scratch_store=scratch_store,
            tasks=tasks,
        )

    for dataset_id in normalize_ids:
        dataset = get_dataset(dataset_id)
        _require_ingested_manifest(dataset, store, parser)
        dataset.normalize(
            store,
            store,
            normalize_config,
            scratch_store=scratch_store,
        )


if __name__ == "__main__":
    main()
