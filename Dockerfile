# syntax=docker/dockerfile:1
#
# Conveyor runs two runtimes at once: the Python evolver graph, and one Node sidecar per painting running
# Pi. Aphrodite has Python 3.10 and Node 16; conveyor needs Python >= 3.11 and pi-agent-core needs modern
# Node, so both come from the image instead of the host.

FROM node:20-bookworm-slim AS node

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# git: the darwinian_evolver dependency is a git+https reference resolved at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

WORKDIR /app

# Dependencies before source, so editing a .py file doesn't reinstall numpy or re-clone darwinian_evolver.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY pi-painter/package.json pi-painter/package-lock.json ./pi-painter/
RUN cd pi-painter && npm ci

COPY src ./src
COPY pi-painter/painter.mjs ./pi-painter/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

# pi_harness derives PI_DIR from parents[3] of its own file, so the sidecar is only found while the package
# stays laid out as /app/src/conveyor with an editable install. Fail the build here rather than at the
# first painting if that ever stops holding.
RUN python -c "\
from conveyor.painting.pi_harness import PI_DIR, PI_SCRIPT; \
from conveyor.painting.canvas import default_targets; \
assert PI_SCRIPT.is_file(), PI_SCRIPT; \
assert (PI_DIR / 'node_modules' / '@earendil-works' / 'pi-agent-core').is_dir(), PI_DIR; \
train, holdout = default_targets(); \
print('layout ok:', PI_DIR, [t.name for t in train + holdout])"

# The event log lives on a volume; compose mounts over this.
RUN mkdir -p /data

CMD ["conveyor", "--help"]
