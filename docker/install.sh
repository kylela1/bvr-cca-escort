#!/usr/bin/env bash
# Installation script for the bvr-cca-escort Docker image.
#
# Run this once per workspace/machine to build the image and confirm it
# actually works (Python deps install + the exported Godot binary launches
# and connects over its local socket). Safe to re-run any time -- Docker
# layer caching keeps re-builds fast.
#
# Usage:
#   docker/install.sh                  # build + smoke test
#   docker/install.sh --no-smoke-test  # build only
#   GODOT_CI_TAG=4.5.1 docker/install.sh   # pin a different Godot version
set -euo pipefail

# On Windows, Git Bash/MSYS auto-converts POSIX-looking paths in command-line
# arguments before they reach docker.exe -- this silently mangles the -v
# host:container volume flags below (both sides get corrupted, and Docker
# just creates/mounts an empty directory instead of erroring). Harmless no-op
# on real Linux/macOS bash.
export MSYS_NO_PATHCONV=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="${IMAGE_NAME:-bvr-cca-escort}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
GODOT_CI_TAG="${GODOT_CI_TAG:-4.5.2}"
RUN_SMOKE_TEST=1

for arg in "$@"; do
  case "$arg" in
    --no-smoke-test) RUN_SMOKE_TEST=0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not on PATH. Install Docker Desktop (or the Docker Engine) first." >&2
  exit 1
fi

echo "==> Building ${IMAGE_NAME}:${IMAGE_TAG} (Godot ${GODOT_CI_TAG})"
docker build \
  --build-arg GODOT_CI_TAG="${GODOT_CI_TAG}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .

mkdir -p results

if [ "$RUN_SMOKE_TEST" -eq 1 ]; then
  echo "==> Running unit tests (pytest -m 'not slow')"
  docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash -lc "pytest -m 'not slow' -q"

  echo "==> Running a short end-to-end smoke test (confirms Godot launches and trains)"
  docker run --rm -v "${REPO_ROOT}/results:/workspace/results" "${IMAGE_NAME}:${IMAGE_TAG}" \
    bash -lc "python src/rl_files/train_bace_clean.py --total-timesteps 512 --n-steps 256 --batch-size 64 --epochs 1 --run-name docker_smoketest"

  rm -rf "${REPO_ROOT}/results/runs_sb3/docker_smoketest"
  echo "==> Smoke test passed."
fi

cat <<EOF

==> Done. Image ready: ${IMAGE_NAME}:${IMAGE_TAG}

Run a real training job (results land in ./results on the host):
  docker run --rm -v "${REPO_ROOT}/results:/workspace/results" ${IMAGE_NAME}:${IMAGE_TAG} \\
    python src/rl_files/train_bace_clean.py --total-timesteps 10000000 --experiment-name my_run

Run evaluation against a completed run:
  docker run --rm -v "${REPO_ROOT}/results:/workspace/results" ${IMAGE_NAME}:${IMAGE_TAG} \\
    python src/rl_files/evaluate_experiment_seeds.py --experiment-dir results/runs_sb3/my_run --episodes 200

Drop into a shell for anything else:
  docker run --rm -it -v "${REPO_ROOT}/results:/workspace/results" ${IMAGE_NAME}:${IMAGE_TAG} bash
EOF
