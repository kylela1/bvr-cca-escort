# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: export the Linux build of the Godot simulator (bin/B_ACE_v0.1.x86_64)
#
# This runs the same two steps as docs/HOW_TO_RUN.md "Full Setup From Scratch"
# steps 2 and 5, but headless and for the "Linux" preset instead of "Windows
# Desktop" -- that preset already exists in Godot_Air_Combat/export_presets.cfg,
# so nothing in the Godot project needs to change.
#
# barichello/godot-ci tags track upstream Godot release versions exactly and
# ship the matching export templates baked in. Keep GODOT_CI_TAG equal to the
# Godot version this project targets (see Godot_Air_Combat/project.godot's
# config/features, currently 4.5.x; docs/HOW_TO_RUN.md used 4.5.2).
# ---------------------------------------------------------------------------
ARG GODOT_CI_TAG=4.5.2
FROM barichello/godot-ci:${GODOT_CI_TAG} AS godot-export

WORKDIR /repo
COPY Godot_Air_Combat/ ./Godot_Air_Combat/

RUN mkdir -p bin \
    && godot --headless --path Godot_Air_Combat --import --quit || true \
    && godot --headless --path Godot_Air_Combat --import --quit \
    && godot --headless --path Godot_Air_Combat --export-release "Linux" /repo/bin/B_ACE_v0.1.x86_64 \
    && test -f /repo/bin/B_ACE_v0.1.x86_64 \
    && test -f /repo/bin/B_ACE_v0.1.pck \
    && chmod +x /repo/bin/B_ACE_v0.1.x86_64

# ---------------------------------------------------------------------------
# Stage 2: the actual training/evaluation runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Shared libraries the exported Godot binary dynamically links against.
# Training itself runs with --headless (no window, no GPU needed -- see
# src/b_ace_py/godot_env.py), but the binary still needs these .so files
# present to start at all. If it fails to launch with a missing-library
# error, add the missing package to this list.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxcursor1 libxinerama1 libxi6 libxrandr2 libxrender1 libxext6 \
    libgl1 libglu1-mesa libasound2 libpulse0 libdbus-1-3 libudev1 \
    fontconfig ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/
COPY pytest.ini .
COPY --from=godot-export /repo/bin/ ./bin/

# Results are meant to be bind-mounted from the host so they survive past the
# container's lifetime -- see docker/install.sh.
RUN mkdir -p results

CMD ["bash"]
