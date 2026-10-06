# syntax=docker/dockerfile:1
#
# Conveyor runs the Python evolver and, per Pi session, a Node sidecar (pi-agent/agent.mjs). Aphrodite has
# Python 3.10 and Node 16; conveyor needs Python >= 3.11 and Pi needs Node 22.19+, so both come from the image.
# Claude Code is installed too, but not logged in: log in once inside the running container (see compose.yml).

FROM node:22-bookworm-slim AS node

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

# The Claude Code CLI for the claude harness. The login is not baked in; it lives on the /data volume.
RUN npm install -g @anthropic-ai/claude-code && claude --version

WORKDIR /app

# Dependencies before source, so editing a .py file doesn't reinstall numpy.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY pi-agent/package.json pi-agent/package-lock.json ./pi-agent/
RUN cd pi-agent && npm ci

COPY src ./src
COPY pi-agent/agent.mjs pi-agent/auth.mjs pi-agent/credentials.mjs pi-agent/painter-context.mjs pi-agent/transient.mjs ./pi-agent/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

# conveyor.pi finds the sidecar as parents[2] of its own file, so it only works while the package stays laid out
# as /app/src/conveyor with an editable install. Fail the build here rather than at the first painting.
RUN python -c "\
from conveyor.pi import PI_DIR, PI_SCRIPT, available; \
from conveyor.painting.canvas import TARGETS_DIR; \
assert PI_SCRIPT.is_file(), PI_SCRIPT; \
assert (PI_DIR / 'painter-context.mjs').is_file(); \
assert available() is None, available(); \
from conveyor.claude import available as claude_available; \
assert claude_available(), 'claude is not on PATH'; \
assert (TARGETS_DIR / 'self_portrait.jpg').is_file(); \
print('layout ok:', PI_DIR)"

# The database and session files live on a volume; compose mounts over this.
RUN mkdir -p /data

CMD ["conveyor", "--help"]
