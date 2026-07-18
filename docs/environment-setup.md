# Environment Setup

Container work happens as `ubuntu`. `root` is only needed for provider bootstrapping
or host-level fixes. `/usr/local/bin/container-init` prepares `/home/ubuntu`,
`/workspace/ubuntu`, and `/data`, then normal project setup should run as `ubuntu`.

The dev image uses Homebrew/Linuxbrew for broad tool availability across local Docker,
Vast.ai, and RunPod. Production images can branch before that layer and install only
required tools with `apt`, `mise`, or another native package manager.

## Shared Setup

Create the project and optional-dotfiles environment files:

```sh
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env
```

Important values:

- `HOST_DATA_ROOT` is mounted from the host into the container at `/data`.
- `DATA_ROOT` is loaded by `uv run` from `.env`; it is not exported into shells by default.
- `PROJECT_KEY` defaults to `batgrad` and controls `/workspace/ubuntu/$PROJECT_KEY`.
- `HOST_UID` and `HOST_GID` map the local Compose container user to the host user.
- `SSH_AUTH_SOCK_HOST` points Compose at the host SSH agent socket.
- `UV_ENV_FILE=.env` lets `uv run` load the project env file. Existing shell variables
  take precedence, so do not export `DATA_ROOT` manually unless you want to override `.env`.

Dotfiles remain disabled by default. If wanted, configure `docker/dotfiles.env` before running
setup. Both env files are sourced as shell code, so only use trusted contents and restrict access
with `chmod 600 .env docker/dotfiles.env` when they contain secrets.

After entering the local container or remote instance, run the repo setup:

```sh
./scripts/setup_project.sh
```

This installs or updates Homebrew tools, optionally runs dotfiles setup, creates the
`uv` environment, runs `uv sync --all-groups`, and installs Playwright Chromium when
Playwright is present. The no-flag command updates the image's Homebrew packages; enabled dotfiles
should therefore use an idempotent setup command. Use `./scripts/setup_project.sh --help` for
focused modes.

## Local Docker Compose

Copy both files:

```sh
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env
```

Set `HOST_DATA_ROOT` in `.env` and optionally configure trusted dotfiles before starting Compose.
File mode `600` is optional but recommended when either file contains secrets. Then continue:

```sh
docker compose up -d --build dev
docker compose exec -it dev zsh
./scripts/setup_project.sh
```

When dotfiles are enabled before Compose starts, container initialization runs their setup once;
the no-flag setup command runs the configured dotfiles command again, so it must be safe to rerun.

## Remote Providers

Vast.ai and RunPod use the same remote image layout and init logic. Build the provider
target, start the instance, SSH as `ubuntu`, clone the repo, copy or mount data at
`/data`, then run the shared setup.

### Vast.ai

```sh
docker buildx build -f docker/Dockerfile --platform linux/amd64 --target remote-vastai \
  --build-arg BASE_IMAGE=vastai/base-image:cuda-13.0.2-cudnn-devel-ubuntu24.04-2026-03-26 \
  -t <your-docker-registry> --push .
```

### RunPod

```sh
docker buildx build -f docker/Dockerfile --platform linux/amd64 --target remote-runpod \
  --build-arg BASE_IMAGE=runpod/base:1.0.6-dev-feat-sonarqube-cuda1300-ubuntu2404 \
  -t <your-docker-registry> --push .
```

## Remote Instance Setup

```sh
ssh -A -L 2718:localhost:2718 -p <port> ubuntu@<host>
```

Keep provider-supplied identity options and replace only `root@<host>` with `ubuntu@<host>`.
Omit `-A` unless private dotfiles or dependencies require agent forwarding on a trusted instance.
After connecting:

```sh
cd /workspace/ubuntu
git clone https://github.com/marplan/batgrad.git
cd batgrad
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env
```

Review `.env` without changing `PROJECT_KEY=batgrad`. If wanted, enable and configure trusted
dotfiles in `docker/dotfiles.env`. File mode `600` is optional but recommended if either file
contains secrets. Then continue:

```sh
./scripts/setup_project.sh
```

Continue with the asset-download and Marimo commands in the [Quick Start](quick-start.md).
