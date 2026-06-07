from __future__ import annotations

from typing import TYPE_CHECKING

from batgrad.data.datasets.pozzato_2022.raw import Pozzato2022RawAdapter
from batgrad.data.datasets.pozzato_2022.specs import DATASET_SPEC
from batgrad.data.processing.raw import RawProcessor

if TYPE_CHECKING:
    from batgrad.data.processing.config import NormalizeStageConfig, RawStageConfig
    from batgrad.storage.store import DataStore


class Pozzato2022Dataset:
    spec = DATASET_SPEC

    def raw_to_parquet(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: RawStageConfig,
    ) -> None:
        RawProcessor(Pozzato2022RawAdapter(self.spec)).run(
            input_store=input_store,
            output_store=output_store,
            config=config,
        )

    def normalize(
        self,
        input_store: DataStore,
        output_store: DataStore,
        config: NormalizeStageConfig,
    ) -> None: ...
