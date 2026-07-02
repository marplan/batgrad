from __future__ import annotations


def dataset_id_from_manifest_path(manifest_path: str) -> str:
    for part in manifest_path.split("/"):
        if part.startswith("dataset="):
            return part.removeprefix("dataset=")
    return ""
