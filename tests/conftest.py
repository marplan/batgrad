from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures import store_at


@pytest.fixture
def local_store(tmp_path: Path):
    return store_at(tmp_path / "store")
