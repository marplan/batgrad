from __future__ import annotations

import tempfile

from batgrad.data.datasets.registry import get_dataset
from batgrad.data.processing.normalize import NormalizeStageConfig
from batgrad.data.processing.raw import IngestStageConfig
from batgrad.logging import configure_logger
from batgrad.storage.local import LocalDataProcessingStore

if __name__ == "__main__":
    configure_logger(level="INFO")
    for dataset_id in ["pozzato-2022", "synthetic-pozzato-2022-m50t"]:
        dataset = get_dataset(dataset_id)
        input_store = LocalDataProcessingStore("/data/loc_datasets/")
        scratch_store = LocalDataProcessingStore(tempfile.gettempdir())
        dataset.ingest(
            input_store,
            input_store,
            IngestStageConfig(
                n_jobs=-1,
            ),
            scratch_store=scratch_store,
        )
        dataset.normalize(
            input_store,
            input_store,
            NormalizeStageConfig(
                n_jobs=-1,
            ),
            scratch_store=scratch_store,
        )
