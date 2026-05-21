#!/usr/bin/env bash

set -Eeuo pipefail

export CONTAINER_INIT_WAIT_FOR_SSH_KEYS="${CONTAINER_INIT_WAIT_FOR_SSH_KEYS:-1}"

/usr/local/bin/container-init /bin/true
