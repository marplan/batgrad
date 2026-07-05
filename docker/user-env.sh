#!/usr/bin/env sh

if [ "$(id -u)" -eq 0 ]; then
    HOME=/root
else
    HOME=/home/ubuntu
fi

export HOME
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_DATA_HOME="$HOME/.local/share"
export UV_PYTHON_INSTALL_DIR="$XDG_DATA_HOME/uv/python"

export PROJECT_KEY="${PROJECT_KEY:-batgrad}"
export UV_ENV_FILE="${UV_ENV_FILE:-.env}"
PROJECT_ROOT="/workspace/ubuntu/$PROJECT_KEY"

if [ "${UV_ENV_FILE#/}" = "$UV_ENV_FILE" ] && [ -d "$PROJECT_ROOT" ]; then
    UV_ENV_FILE="$PROJECT_ROOT/$UV_ENV_FILE"
fi

if [ "$(id -u)" -ne 0 ]; then
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/$PROJECT_KEY}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_PROJECT_ENVIRONMENT/.uv-cache}"
    umask 0002
fi

export HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-/home/linuxbrew/.linuxbrew}"
if [ "$(id -u)" -ne 0 ] && [ -x "$HOMEBREW_PREFIX/bin/brew" ]; then
    eval "$("$HOMEBREW_PREFIX/bin/brew" shellenv)"
fi

if [ -x /usr/local/cuda/bin/nvcc ]; then
    case ":${PATH:-}:" in
        *:/usr/local/cuda/bin:*) ;;
        *) PATH="/usr/local/cuda/bin${PATH:+:$PATH}" ;;
    esac
fi

export PATH
export UV_PROJECT_ENVIRONMENT
export UV_CACHE_DIR
