from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from batgrad.contracts.mapping import BaseColumns, DatasetStageId
from batgrad.contracts.segments import ParquetSegment, segment_values
from batgrad.data.datasets.registry import DatasetId, dataset_ids, get_dataset
from batgrad.data.processing.manifests import load_stage_manifest
from batgrad.logging import configure_logging, get_logger
from batgrad.ml.checkpoint import export_checkpoint
from batgrad.storage.local import LocalDataProcessingStore

logger = get_logger("batgrad.scripts.hf_assets")

HF_REPO_ID = "marplan6/batgrad"
DEFAULT_CHECKPOINT = "init_baseline"
CHECKPOINTS = (DEFAULT_CHECKPOINT,)
BYTE_UNIT = 1024.0


def _add_selectors(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=dataset_ids(),
        metavar="DATASET_ID",
        nargs="+",
        help="registered normalized datasets to transfer (download default: all)",
    )
    parser.add_argument(
        "--ckpt",
        choices=CHECKPOINTS,
        metavar="CHECKPOINT_ID",
        help="checkpoint to transfer (download default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="dataset store root (default: DATA_ROOT)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transfer batgrad Hugging Face assets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download selected assets")
    _add_selectors(download)
    download.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="model artifact root (default: outputs)",
    )

    upload = subparsers.add_parser("upload", help="upload selected assets")
    _add_selectors(upload)
    upload.add_argument(
        "--checkpoint-path",
        type=Path,
        help="training checkpoint to compact and upload with --ckpt",
    )
    upload.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and list assets without uploading",
    )
    return parser


def _selected_datasets(args: argparse.Namespace) -> tuple[DatasetId, ...]:
    selected = args.dataset
    if selected is None and args.ckpt is None:
        return dataset_ids()
    return tuple(dict.fromkeys(selected or ()))


def _data_root(value: Path | None, parser: argparse.ArgumentParser) -> Path:
    raw = str(value) if value is not None else os.getenv("DATA_ROOT")
    if raw is None or not raw.strip():
        parser.error("--data-root is required when DATA_ROOT is not set")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        parser.error(f"data root must be absolute: {root}")
    return root


def _dataset_prefix(dataset_id: DatasetId) -> str:
    return get_dataset(dataset_id).spec.source_root(DatasetStageId.normalized)


def _dataset_patterns(selected: tuple[DatasetId, ...]) -> tuple[str, ...]:
    return tuple(f"{_dataset_prefix(dataset_id)}/**" for dataset_id in selected)


def _dataset_upload_files(root: Path, selected: tuple[DatasetId, ...]) -> tuple[str, ...]:
    store = LocalDataProcessingStore(root)
    files: set[str] = set()
    for dataset_id in selected:
        dataset = get_dataset(dataset_id)
        manifest_path = dataset.spec.manifest(DatasetStageId.normalized)
        manifest = load_stage_manifest(dataset.spec, store, DatasetStageId.normalized)
        if str(BaseColumns.norm_segs) not in manifest.columns:
            raise ValueError(f"Normalized manifest has no segment column: {manifest_path}")
        files.add(manifest_path)
        prefix = f"{_dataset_prefix(dataset_id)}/"
        for value in manifest[str(BaseColumns.norm_segs)].to_list():
            for segment_value in segment_values(value):
                segment = ParquetSegment.from_value(segment_value)
                if not segment.path.startswith(prefix):
                    raise ValueError(f"Normalized segment is outside dataset root: {segment.path}")
                if not Path(store.resolve(segment.path)).is_file():
                    raise FileNotFoundError(f"Missing normalized segment: {segment.path}")
                files.add(segment.path)
    return tuple(sorted(files))


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < BYTE_UNIT or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= BYTE_UNIT
    raise AssertionError


def _print_files(root: Path, files: tuple[str, ...]) -> None:
    total = 0
    for location in files:
        size = (root / location).stat().st_size
        total += size
        logger.info("asset path=%s size=%s", location, _format_bytes(size))
    logger.info("asset total files=%d size=%s", len(files), _format_bytes(total))


def _warn_without_xet() -> None:
    if importlib.util.find_spec("hf_xet") is None:
        logger.warning("hf_xet is unavailable; Hugging Face transfers will use the legacy backend")


def _download(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    selected = _selected_datasets(args)
    checkpoints = CHECKPOINTS if args.dataset is None and args.ckpt is None else (args.ckpt,)
    if selected:
        root = _data_root(args.data_root, parser)
        root.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading datasets=%s to %s", ",".join(selected), root)
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=root,
            allow_patterns=list(_dataset_patterns(selected)),
        )
    for checkpoint_id in (checkpoint for checkpoint in checkpoints if checkpoint is not None):
        outputs_root = args.outputs_root.expanduser()
        outputs_root.mkdir(parents=True, exist_ok=True)
        pattern = f"checkpoints/{checkpoint_id}.pt"
        logger.info("Downloading checkpoint=%s to %s", checkpoint_id, outputs_root)
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="model",
            local_dir=outputs_root,
            allow_patterns=[pattern],
        )


def _upload_datasets(
    api: HfApi,
    root: Path,
    selected: tuple[DatasetId, ...],
    *,
    dry_run: bool,
) -> None:
    files = _dataset_upload_files(root, selected)
    _print_files(root, files)
    if dry_run:
        return
    api.auth_check(repo_id=HF_REPO_ID, repo_type="dataset", write=True)
    api.upload_folder(
        folder_path=root,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=list(files),
        ignore_patterns=["**/.DS_Store", "**/.cache/**", "**/scratch/**"],
        commit_message="Upload normalized batgrad baseline datasets",
    )


def _upload_checkpoint(
    api: HfApi,
    checkpoint_id: str,
    source: Path,
    *,
    dry_run: bool,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    with tempfile.TemporaryDirectory(prefix="batgrad-hf-") as directory:
        exported = Path(directory) / "final.pt"
        export_checkpoint(source, exported)
        size = exported.stat().st_size
        location = f"checkpoints/{checkpoint_id}.pt"
        logger.info("asset path=%s size=%s", location, _format_bytes(size))
        if dry_run:
            return
        api.auth_check(repo_id=HF_REPO_ID, repo_type="model", write=True)
        api.upload_file(
            path_or_fileobj=exported,
            path_in_repo=location,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Upload {checkpoint_id} checkpoint",
        )


def _upload(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    selected = _selected_datasets(args)
    if args.ckpt is None and args.checkpoint_path is not None:
        parser.error("--checkpoint-path requires --ckpt")
    if args.ckpt is not None and args.checkpoint_path is None:
        parser.error("--ckpt requires --checkpoint-path for upload")
    api = HfApi()
    if selected:
        _upload_datasets(
            api,
            _data_root(args.data_root, parser),
            selected,
            dry_run=args.dry_run,
        )
    if args.ckpt is not None:
        _upload_checkpoint(
            api,
            args.ckpt,
            args.checkpoint_path.expanduser(),
            dry_run=args.dry_run,
        )


def main() -> None:
    configure_logging()
    _warn_without_xet()
    parser = _parser()
    args = parser.parse_args()
    if args.command == "download":
        _download(args, parser)
    else:
        _upload(args, parser)


if __name__ == "__main__":
    main()
