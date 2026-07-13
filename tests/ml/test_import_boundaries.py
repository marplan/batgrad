from __future__ import annotations

import ast
from pathlib import Path


def test_ml_does_not_import_data_package_or_notebooks() -> None:
    for path in Path("batgrad/ml").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _imports(tree)
        assert not any(
            name == "batgrad.data" or name.startswith("batgrad.data.") for name in imports
        )
        assert not any(name == "notebooks" or name.startswith("notebooks.") for name in imports)


def test_numerical_ml_does_not_import_data_or_presentation() -> None:
    forbidden = (
        "batgrad.ml.data",
        "batgrad.ml.loggers",
        "batgrad.storage",
        "batgrad.viz",
    )
    for path in (
        "batgrad/ml/loss.py",
        "batgrad/ml/masked_suffix.py",
        "batgrad/ml/nn.py",
        "batgrad/ml/objective.py",
        "batgrad/ml/rollout.py",
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        assert not any(name.startswith(forbidden) for name in _imports(tree))


def test_core_use_cases_do_not_depend_on_presentation_interfaces() -> None:
    forbidden_by_path = {
        "batgrad/ml/config.py": ("batgrad.ml.loggers",),
        "batgrad/ml/validation.py": ("batgrad.ml.loggers", "batgrad.viz"),
        "batgrad/ml/checkpoint.py": ("batgrad.ml.loggers",),
    }
    for path, forbidden in forbidden_by_path.items():
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        assert not any(name.startswith(forbidden) for name in _imports(tree))


def test_ml_visualization_does_not_plan_or_materialize_batches() -> None:
    tree = ast.parse(Path("batgrad/viz/ml.py").read_text(encoding="utf-8"))
    assert not any(name.startswith("batgrad.ml.data.materialization") for name in _imports(tree))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint({"iter_batch_plans", "materialize_batch_plan"})


def test_inference_notebook_does_not_implement_model_execution() -> None:
    forbidden_imports = (
        "torch",
        "batgrad.ml.checkpoint",
        "batgrad.ml.data.materialization",
        "batgrad.ml.data.planning",
        "batgrad.ml.nn",
        "batgrad.ml.rollout",
    )
    for path in ("notebooks/inference.py", "notebooks/inference_helpers.py"):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        assert not any(name.startswith(forbidden_imports) for name in _imports(tree))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "load" not in calls
        assert "load_state_dict" not in calls


def test_ml_notebooks_do_not_import_numerical_execution_modules() -> None:
    forbidden_imports = (
        "torch",
        "batgrad.ml.checkpoint",
        "batgrad.ml.loss",
        "batgrad.ml.masked_suffix",
        "batgrad.ml.nn",
        "batgrad.ml.objective",
        "batgrad.ml.rollout",
        "batgrad.ml.validation",
    )
    for path in (
        "notebooks/createconfig.py",
        "notebooks/inference.py",
        "notebooks/inference_helpers.py",
        "notebooks/ml.py",
        "notebooks/ml_helpers.py",
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        assert not any(name.startswith(forbidden_imports) for name in _imports(tree))


def _imports(tree: ast.AST) -> tuple[str, ...]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return tuple(names)
