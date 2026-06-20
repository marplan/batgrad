from __future__ import annotations

from batgrad.data.datasets.registry import get_dataset
from batgrad.data.processing.raw import IngestStageConfig
from batgrad.logging import configure
from batgrad.storage.local import LocalDataProcessingStore

if __name__ == "__main__":
    configure(level="INFO")
    dataset = get_dataset("pozzato-2022")
    input_store = LocalDataProcessingStore("/data/loc_datasets/")
    dataset.ingest(
        input_store,
        input_store,
        IngestStageConfig(
            n_jobs=-1,
        ),
    )
