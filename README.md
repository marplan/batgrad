## Development Setup

Start the local container with Docker Compose:

```sh
docker compose up -d --build
```

Run the repo-local setup from inside the container or a remote instance:

```sh
./scripts/setup_project.sh
```

Useful focused modes:

```sh
./scripts/setup_project.sh --tools-only
./scripts/setup_project.sh --dotfiles-only
./scripts/setup_project.sh --project-only
```

Dotfiles are not baked into the image. For optional personal dotfiles setup, copy
`docker/dotfiles.env.example` to `docker/dotfiles.env`. If `ENABLE_DOTFILES=1`, setup
clones or reuses `DOTFILES_DIR` and runs `DOTFILES_SETUP_CMD` from that directory.
