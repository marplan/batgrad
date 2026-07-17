# Environment Setup

Container work happens as `ubuntu`. `root` is only needed for provider bootstrapping
or host-level fixes. `/usr/local/bin/container-init` prepares `/home/ubuntu`,
`/workspace/ubuntu`, and `/data`, then normal project setup should run as `ubuntu`.

The dev image uses Homebrew/Linuxbrew for broad tool availability across local Docker,
Vast.ai, and RunPod. Production images can branch before that layer and install only
required tools with `apt`, `mise`, or another native package manager.

## Shared Setup

Create `.env` from the example:

```sh
cp .env.example .env
```

Important values:

- `HOST_DATA_ROOT` is mounted from the host into the container at `/data`.
- `DATA_ROOT` is loaded by `uv run` from `.env`; it is not exported into shells by default.
- `PROJECT_KEY` defaults to `batgrad` and controls `/workspace/ubuntu/$PROJECT_KEY`.
- `HOST_UID` and `HOST_GID` map the local Compose container user to the host user.
- `SSH_AUTH_SOCK_HOST` points Compose at the host SSH agent socket.
- `UV_ENV_FILE=.env` lets `uv run` load the project env file. Existing shell variables
  take precedence, so do not export `DATA_ROOT` manually unless you want to override `.env`.

Optional dotfiles use a separate env file:

```sh
cp docker/dotfiles.env.example docker/dotfiles.env
```

After entering the local container or remote instance, run the repo setup:

```sh
./scripts/setup_project.sh
```

This installs or updates Homebrew tools, optionally runs dotfiles setup, creates the
`uv` environment, runs `uv sync --all-groups`, and installs Playwright Chromium when
Playwright is present. Use `./scripts/setup_project.sh --help` for focused modes.

## Local Docker Compose

```sh
docker compose up -d --build dev
docker compose exec -it dev zsh
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env  # optional
./scripts/setup_project.sh
```

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
ssh ubuntu@<host>
git clone <repo-url> /workspace/ubuntu/batgrad
cd /workspace/ubuntu/batgrad
cp .env.example .env
cp docker/dotfiles.env.example docker/dotfiles.env  # optional
./scripts/setup_project.sh
```
