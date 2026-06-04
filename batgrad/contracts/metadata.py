from __future__ import annotations

from dataclasses import dataclass

from batgrad.contracts.columns import BaseColumns, ColumnSpec, MetadataColumns


@dataclass(frozen=True, slots=True)
class MetadataLayout:
    manifest: tuple[ColumnSpec, ...] = (
        BaseColumns.dataset_id,
        MetadataColumns.chem,
        MetadataColumns.nom_capa,
    )
    footer: tuple[ColumnSpec, ...] = (
        BaseColumns.dataset_id,
        MetadataColumns.chem,
    )
