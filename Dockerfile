ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

FROM ${BASE_IMAGE} AS common

ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV HOMEBREW_PREFIX=/home/linuxbrew/.linuxbrew

RUN <<EOF
set -e
rm -f /etc/dpkg/dpkg.cfg.d/excludes
if dpkg-divert --list /usr/bin/man >/dev/null 2>&1; then
    dpkg-divert --remove --rename /usr/bin/man
fi
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    file \
    git \
    less \
    libsundials-dev \
    libsuitesparse-dev \
    man-db \
    manpages \
    manpages-dev \
    ncurses-bin \
    openssh-client \
    procps \
    python3 \
    python3-venv \
    sudo \
    unzip \
    zsh
rm -rf /var/lib/apt/lists/*
EOF

RUN <<EOF
set -e
if ! getent group ubuntu >/dev/null 2>&1; then
    groupadd --gid ${USER_GID} ubuntu
fi
if ! id ubuntu >/dev/null 2>&1; then
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /usr/bin/zsh ubuntu
fi
usermod -s /usr/bin/zsh ubuntu
printf '%s\n' 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu
printf '%s\n' 'Defaults env_keep += "SSH_AUTH_SOCK TERM COLORTERM"' >> /etc/sudoers.d/ubuntu
chmod 0440 /etc/sudoers.d/ubuntu
install -d -m 0755 -o ubuntu -g ubuntu /workspace /workspace/ubuntu
install -d -m 0755 -o ubuntu -g ubuntu /home/linuxbrew
install -d -m 0775 -o ubuntu -g ubuntu /data
EOF

COPY docker/user-env.sh /etc/profile.d/dev-env.sh
COPY docker/container-init.sh /usr/local/bin/container-init
COPY docker/brew-packages.txt /tmp/brew-packages.txt
COPY docker/terminfo/xterm-ghostty.terminfo /tmp/xterm-ghostty.terminfo

RUN <<EOF
set -e
chmod 0755 /usr/local/bin/container-init
tic -x -o /usr/share/terminfo /tmp/xterm-ghostty.terminfo
rm -f /tmp/xterm-ghostty.terminfo
loader='[ -f /etc/profile.d/dev-env.sh ] && . /etc/profile.d/dev-env.sh'
printf '%s\n' "$loader" >> /etc/profile
printf '%s\n' "$loader" >> /etc/bash.bashrc
install -d -m 0755 /etc/zsh
printf '%s\n' "$loader" >> /etc/zsh/zshenv
printf '%s\n' "$loader" >> /etc/zsh/zprofile
printf '%s\n' "$loader" >> /etc/zsh/zshrc
EOF

USER ubuntu
WORKDIR /workspace/ubuntu

RUN <<EOF
set -e
mkdir -p \
    /home/ubuntu/.cache \
    /home/ubuntu/.config \
    /home/ubuntu/.local/bin \
    /home/ubuntu/.local/share \
    /home/ubuntu/.venvs
touch /home/ubuntu/.zshrc
EOF

RUN <<EOF
set -e
export HOME=/home/ubuntu
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_ANALYTICS=1
export HOMEBREW_NO_ENV_HINTS=1
mkdir -p "${HOMEBREW_PREFIX}/bin"
git clone --depth=1 https://github.com/Homebrew/brew "${HOMEBREW_PREFIX}/Homebrew"
ln -sf ../Homebrew/bin/brew "${HOMEBREW_PREFIX}/bin/brew"
eval "$("${HOMEBREW_PREFIX}/bin/brew" shellenv)"
brew install $(grep -v '^[[:space:]]*$' /tmp/brew-packages.txt)
brew cleanup --prune=all
EOF

FROM common AS dev

USER ubuntu
WORKDIR /workspace/ubuntu
ENTRYPOINT ["/usr/local/bin/container-init"]
CMD ["zsh"]

FROM common AS remote

USER root
WORKDIR /workspace

FROM remote AS remote-runpod

ENV CONTAINER_INIT_SKIP_WORKSPACE_CHOWN=1

COPY docker/providers/runpod/container-init-hook.sh /post_start.sh
RUN chmod 0755 /post_start.sh

FROM remote AS remote-vastai

COPY docker/providers/vastai/container-init-hook.sh /etc/vast_boot.d/80-container-init-hook.sh
RUN chmod 0755 /etc/vast_boot.d/80-container-init-hook.sh
