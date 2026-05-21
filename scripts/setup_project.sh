#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
USER_ENV_SCRIPT="$PROJECT_ROOT/docker/user-env.sh"
DOTFILES_ENV_FILE="$PROJECT_ROOT/docker/dotfiles.env"
BREW_PACKAGE_FILE="$PROJECT_ROOT/docker/brew-packages.txt"
ORIGINAL_ARGS=("$@")

load_env_file() {
    local env_file="$1"
    if [ -f "$env_file" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$env_file"
        set +a
    fi
}

load_env_file "$PROJECT_ROOT/.env"
load_env_file "$DOTFILES_ENV_FILE"

RUN_TOOLS=1
RUN_DOTFILES=1
RUN_PROJECT=1
RUN_PLAYWRIGHT=1

usage() {
    cat <<'EOF'
Usage: scripts/setup_project.sh [options]

Options:
  --tools-only        Install/update Homebrew packages from docker/brew-packages.txt
  --dotfiles-only     Clone/use dotfiles and run DOTFILES_SETUP_CMD
  --project-only      Create/link the uv environment and run uv sync --all-groups
  --skip-tools        Skip Homebrew package updates
  --skip-dotfiles     Skip dotfiles setup
  --skip-playwright   Skip Playwright Chromium install
  -h, --help          Show this help
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tools-only)
            RUN_TOOLS=1
            RUN_DOTFILES=0
            RUN_PROJECT=0
            RUN_PLAYWRIGHT=0
            ;;
        --dotfiles-only)
            RUN_TOOLS=0
            RUN_DOTFILES=1
            RUN_PROJECT=0
            RUN_PLAYWRIGHT=0
            ;;
        --project-only)
            RUN_TOOLS=0
            RUN_DOTFILES=0
            RUN_PROJECT=1
            ;;
        --skip-tools)
            RUN_TOOLS=0
            ;;
        --skip-dotfiles)
            RUN_DOTFILES=0
            ;;
        --skip-playwright)
            RUN_PLAYWRIGHT=0
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            printf 'Error: unknown option: %s\n' "$1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [ "$(id -u)" -eq 0 ]; then
    exec sudo -H -u ubuntu env \
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
        bash "$PROJECT_ROOT/scripts/setup_project.sh" "${ORIGINAL_ARGS[@]}"
fi

if [ -f /etc/profile.d/dev-env.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/dev-env.sh
fi

if [ -f "$USER_ENV_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$USER_ENV_SCRIPT"
fi

cd "$PROJECT_ROOT"

load_brew_packages() {
    if [ ! -f "$BREW_PACKAGE_FILE" ]; then
        printf 'Error: missing Homebrew package file: %s\n' "$BREW_PACKAGE_FILE" >&2
        exit 1
    fi

    BREW_PACKAGES=()
    local line
    while IFS= read -r line; do
        case "$line" in
            '' | \#*) continue ;;
            *) BREW_PACKAGES+=("$line") ;;
        esac
    done < "$BREW_PACKAGE_FILE"

    if [ "${#BREW_PACKAGES[@]}" -eq 0 ]; then
        printf 'Error: no Homebrew packages listed in %s\n' "$BREW_PACKAGE_FILE" >&2
        exit 1
    fi
}

ensure_tools() {
    if ! command -v brew >/dev/null 2>&1; then
        printf 'Error: Homebrew is not available. Rebuild the image or check docker/user-env.sh.\n' >&2
        exit 1
    fi

    load_brew_packages

    export HOMEBREW_NO_ANALYTICS=1
    export HOMEBREW_NO_ENV_HINTS=1

    echo '==> Updating Homebrew package metadata...'
    brew update

    echo '==> Installing requested Homebrew packages...'
    brew install "${BREW_PACKAGES[@]}"

    echo '==> Upgrading requested Homebrew packages when outdated...'
    brew upgrade "${BREW_PACKAGES[@]}" || true
}

ensure_dotfiles_known_hosts() {
    case "${DOTFILES_REPO:-}" in
        git@github.com:* | ssh://*github.com/*)
            mkdir -p "$HOME/.ssh"
            touch "$HOME/.ssh/known_hosts"
            chmod 0700 "$HOME/.ssh"
            chmod 0600 "$HOME/.ssh/known_hosts"
            ssh-keygen -F github.com >/dev/null 2>&1 || ssh-keyscan github.com >> "$HOME/.ssh/known_hosts"
            ;;
    esac
}

ensure_dotfiles_checkout() {
    if [ "${ENABLE_DOTFILES:-0}" != "1" ]; then
        echo '==> Dotfiles disabled.'
        return 1
    fi
    if [ -z "${DOTFILES_REPO:-}" ]; then
        printf 'Error: ENABLE_DOTFILES=1 but DOTFILES_REPO is empty.\n' >&2
        exit 1
    fi

    DOTFILES_DIR="${DOTFILES_DIR:-$HOME/dotfiles}"
    DOTFILES_REF="${DOTFILES_REF:-main}"
    export DOTFILES_DIR DOTFILES_REF

    ensure_dotfiles_known_hosts

    if [ -d "$DOTFILES_DIR/.jj" ] || [ -d "$DOTFILES_DIR/.git" ]; then
        printf '==> Using existing dotfiles checkout: %s\n' "$DOTFILES_DIR"
        return 0
    fi
    if [ -e "$DOTFILES_DIR" ]; then
        printf 'Error: DOTFILES_DIR exists but is not a jj/git checkout: %s\n' "$DOTFILES_DIR" >&2
        exit 1
    fi

    if command -v jj >/dev/null 2>&1; then
        jj git clone --branch "$DOTFILES_REF" "$DOTFILES_REPO" "$DOTFILES_DIR"
    else
        git clone --branch "$DOTFILES_REF" "$DOTFILES_REPO" "$DOTFILES_DIR"
    fi
}

ensure_dotfiles() {
    ensure_dotfiles_checkout || return 0

    if [ -n "${DOTFILES_SETUP_CMD:-}" ]; then
        echo '==> Running DOTFILES_SETUP_CMD...'
        (
            cd "$DOTFILES_DIR"
            export DOTFILES_DIR
            bash -lc "$DOTFILES_SETUP_CMD"
        )
        return
    fi

    printf 'Error: ENABLE_DOTFILES=1 but DOTFILES_SETUP_CMD is not set.\n' >&2
    exit 1
}

ensure_project_environment() {
    if ! command -v uv >/dev/null 2>&1; then
        printf 'Error: uv is not available. Run %s --tools-only first or rebuild the image.\n' "$0" >&2
        exit 1
    fi

    PROJECT_KEY="${PROJECT_KEY:-batgrad}"
    UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/$PROJECT_KEY}"
    UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_PROJECT_ENVIRONMENT/.uv-cache}"
    export PROJECT_KEY UV_PROJECT_ENVIRONMENT UV_CACHE_DIR

    mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$UV_CACHE_DIR"

    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
        printf '==> Creating uv environment: %s\n' "$UV_PROJECT_ENVIRONMENT"
        uv venv --clear "$UV_PROJECT_ENVIRONMENT"
    fi

    if [ -L .venv ] && [ "$(readlink .venv)" != "$UV_PROJECT_ENVIRONMENT" ]; then
        rm -f .venv
    fi
    if [ ! -e .venv ]; then
        ln -s "$UV_PROJECT_ENVIRONMENT" .venv
    elif [ ! -L .venv ]; then
        printf 'Error: .venv exists and is not a symlink. Move it before running setup.\n' >&2
        exit 1
    fi

    echo '==> Syncing uv dependencies...'
    uv sync --all-groups
}

ensure_playwright() {
    if [ "$RUN_PLAYWRIGHT" != "1" ]; then
        return
    fi
    if uv run python -c 'import playwright' >/dev/null 2>&1; then
        echo '==> Installing Playwright Chromium...'
        uv run playwright install --with-deps chromium
    fi
}

if [ "$RUN_TOOLS" = "1" ]; then
    ensure_tools
fi

if [ "$RUN_DOTFILES" = "1" ]; then
    ensure_dotfiles
fi

if [ "$RUN_PROJECT" = "1" ]; then
    ensure_project_environment
    ensure_playwright
fi

echo '==> Setup complete.'
