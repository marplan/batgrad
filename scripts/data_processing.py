from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Literal, cast

from batgrad.data.datasets.registry import DatasetId, dataset_ids, get_dataset
from batgrad.data.processing.normalize import NormalizeStageConfig
from batgrad.data.processing.raw import IngestStageConfig
from batgrad.logging import configure_logging
from batgrad.storage.local import LocalDataProcessingStore

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
        default=Path("/data"),
        help="input and output data store (default: /data)",
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
        dataset.ingest(
            store,
            store,
            ingest_config,
            scratch_store=scratch_store,
        )

    for dataset_id in normalize_ids:
        dataset = get_dataset(dataset_id)
        dataset.normalize(
            store,
            store,
            normalize_config,
            scratch_store=scratch_store,
        )


if __name__ == "__main__":
    main()
