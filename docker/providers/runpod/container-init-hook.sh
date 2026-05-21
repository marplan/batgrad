#!/usr/bin/env bash

set -Eeuo pipefail

log_file=/tmp/batgrad-container-init.log
exec > >(tee -a "$log_file") 2>&1

echo "[batgrad-post-start] starting"
echo "[batgrad-post-start] user=$(id)"

if /usr/local/bin/container-init /bin/true; then
    echo "[batgrad-post-start] workspace layout:"
    ls -ld /workspace /workspace/ubuntu 2>&1 || true
    echo "[batgrad-post-start] done"
else
    code=$?
    echo "[batgrad-post-start] container-init failed with exit code $code"
    echo "[batgrad-post-start] workspace layout:"
    ls -ld /workspace /workspace/ubuntu 2>&1 || true
    exit 0
fi
