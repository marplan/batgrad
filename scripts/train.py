from __future__ import annotations

import argparse

from batgrad.logging import configure_logger
from batgrad.ml.train import train_from_config


def main() -> None:
    configure_logger()
    parser = argparse.ArgumentParser(description="Run a batgrad ML training job")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    train_from_config(args.config)


if __name__ == "__main__":
    main()
