#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_KEY="${PROJECT_KEY:-batgrad}"
PROJECT_ROOT="/workspace/ubuntu/$PROJECT_KEY"
DOTFILES_ENV_FILE="$PROJECT_ROOT/docker/dotfiles.env"

log() {
    printf '[container-init] %s\n' "$*"
}

run_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
        return
    fi

    sudo "$@"
}

run_ubuntu() {
    if [ "$(id -u)" -eq 0 ]; then
        sudo -H -u ubuntu "$@"
        return
    fi

    "$@"
}

wait_for_root_authorized_keys() {
    if [ "${CONTAINER_INIT_WAIT_FOR_SSH_KEYS:-0}" != "1" ]; then
        return
    fi

    local attempts="${CONTAINER_INIT_SSH_WAIT_ATTEMPTS:-60}"
    local delay_seconds="${CONTAINER_INIT_SSH_WAIT_DELAY_SECONDS:-1}"
    local attempt

    for attempt in $(seq 1 "$attempts"); do
        if run_root test -s /root/.ssh/authorized_keys; then
            log "Found /root/.ssh/authorized_keys after ${attempt} check(s)."
            return
        fi
        sleep "$delay_seconds"
    done
}

ensure_layout() {
    run_root install -d -m 0755 -o ubuntu -g ubuntu /home/ubuntu
    run_root mkdir -p /workspace/ubuntu
    if [ "${CONTAINER_INIT_SKIP_WORKSPACE_CHOWN:-0}" != "1" ]; then
        run_root chown ubuntu:ubuntu /workspace/ubuntu
        run_root chmod 0755 /workspace/ubuntu
    fi
    run_root install -d -m 0775 -o ubuntu -g ubuntu "${DATA_ROOT:-/data}" || log "Skipping DATA_ROOT ownership setup."

    if run_root test -d /home/ubuntu/.local; then
        run_root chown ubuntu:ubuntu /home/ubuntu/.local
    fi

    run_ubuntu mkdir -p \
        /home/ubuntu/.cache \
        /home/ubuntu/.config \
        /home/ubuntu/.local/bin \
        /home/ubuntu/.local/share \
        /home/ubuntu/.venvs
    run_ubuntu touch /home/ubuntu/.zshrc
}

sync_ssh_keys() {
    if ! run_root test -s /root/.ssh/authorized_keys; then
        return
    fi

    run_root install -d -m 0700 -o ubuntu -g ubuntu /home/ubuntu/.ssh
    run_root install -m 0600 -o ubuntu -g ubuntu /root/.ssh/authorized_keys /home/ubuntu/.ssh/authorized_keys
}

setup_dotfiles() {
    if [ "${ENABLE_DOTFILES:-0}" != "1" ]; then
        return
    fi

    local setup_script="$PROJECT_ROOT/scripts/setup_project.sh"
    if [ ! -f "$setup_script" ]; then
        log "Skipping dotfiles setup; repo-local setup script is not present at $setup_script."
        return
    fi

    if [ "$(id -u)" -eq 0 ]; then
        sudo -H -u ubuntu env \
            ENABLE_DOTFILES="${ENABLE_DOTFILES:-0}" \
            DOTFILES_REPO="${DOTFILES_REPO:-}" \
            DOTFILES_REF="${DOTFILES_REF:-main}" \
            DOTFILES_DIR="${DOTFILES_DIR:-/home/ubuntu/dotfiles}" \
            DOTFILES_SETUP_CMD="${DOTFILES_SETUP_CMD:-}" \
            DATA_ROOT="${DATA_ROOT:-/data}" \
            PROJECT_KEY="${PROJECT_KEY:-batgrad}" \
            UV_ENV_FILE="${UV_ENV_FILE:-.env}" \
            OP_SERVICE_ACCOUNT_TOKEN="${OP_SERVICE_ACCOUNT_TOKEN:-}" \
            SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-}" \
            bash "$setup_script" --dotfiles-only
        return
    fi

    bash "$setup_script" --dotfiles-only
}

main() {
    wait_for_root_authorized_keys
    ensure_layout
    sync_ssh_keys

    if [ -f /etc/profile.d/dev-env.sh ]; then
        . /etc/profile.d/dev-env.sh
    fi

    if [ -f "$DOTFILES_ENV_FILE" ]; then
        set -a
        . "$DOTFILES_ENV_FILE"
        set +a
    fi

    setup_dotfiles

    if [ "$#" -eq 0 ]; then
        exec zsh
    fi

    exec "$@"
}

main "$@"
