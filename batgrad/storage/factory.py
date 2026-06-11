from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from batgrad.storage.local import LocalDataStore

if TYPE_CHECKING:
    from batgrad.storage.store import DataStore

DATA_ROOT_ENV = "DATA_ROOT"
StorageBackend = Literal["auto", "local"]


@overload
def get_storage(
    root: str | Path,
    backend: Literal["local"],
    *,
    create: bool = False,
) -> LocalDataStore: ...


@overload
def get_storage(
    root: str | Path | None = None,
    backend: Literal["auto"] = "auto",
    *,
    create: bool = False,
) -> DataStore: ...


def get_storage(
    root: str | Path | None = None,
    backend: StorageBackend = "auto",
    *,
    create: bool = False,
) -> DataStore:
    """Create a data store instance. Can be used to create a local or remote data store.

    Args:
        root: The root path to the data store. If not provided, the value of the
            `DATA_ROOT` environment variable is used.
        backend: Storage backend selection. Use `auto` for environment/root-based
            detection or `local` to require an absolute local path.
        create: Create the local root directory when it does not exist.

    Returns:
        A data store instance.

    Raises:
        ValueError: If the data root is incorrect.

    """
    if backend == "local":
        if root is None:
            raise ValueError("backend='local' requires an absolute root path")
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError("backend='local' requires an absolute root path")
        return LocalDataStore(root_path, create=create)

    resolved_root = str(root or os.environ.get(DATA_ROOT_ENV, "")).strip()
    if not resolved_root:
        raise ValueError(f"Data root is not configured. Set {DATA_ROOT_ENV} or pass root.")

    if resolved_root.startswith("/"):
        return LocalDataStore(resolved_root, create=create)

    raise ValueError(f"Unknown data root: {resolved_root}. Provide root as an absolute path.")
