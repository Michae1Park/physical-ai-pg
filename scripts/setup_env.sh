#!/usr/bin/env bash
# Sets up the `openvla` conda env for running OpenVLA against LIBERO. See README.md
# for what/why on each step. Idempotent: safe to re-run.
set -euo pipefail

ENV_NAME="openvla"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already exists, reusing it"
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

# pip's progress-bar rendering segfaults on this box when installing a long dependency
# list (root cause of the torch/openvla install crashes seen during testing)
export PIP_PROGRESS_BAR=off

# belt-and-suspenders: retry with a cache purge in case anything else flaky comes up
pip_install() {
  local attempt=1 max_attempts=3
  until pip install "$@"; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "pip install failed after $max_attempts attempts: $*" >&2
      return 1
    fi
    echo "pip install crashed (attempt $attempt/$max_attempts), purging cache and retrying: $*" >&2
    pip cache purge
    attempt=$((attempt + 1))
  done
}

# pip >= 24's resolver crashes on some of the older pins below
pip_install "pip==23.3.2"

pip_install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

cd "$REPO_ROOT/src/openvla"
pip_install -e .                                                  # OpenVLA + its ML stack
pip_install -r experiments/robot/libero/libero_requirements.txt   # sim stack: robosuite, bddl, ...

cd "$REPO_ROOT/src/LIBERO"
pip_install -e .

# LIBERO's editable install doesn't resolve outside its own dir (namespace package
# quirk) — point a .pth file at the repo root instead
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$REPO_ROOT/src/LIBERO" > "$SITE/zz_libero_root.pth"

# robosuite 1.4.1 segfaults on mujoco >= 3.x
pip_install "numpy<2" "mujoco==2.3.7"

# tensorflow==2.15 requires numpy<2; that drags tensorflow-metadata/wandb down with it
pip_install "tensorflow-metadata==1.14.0"
pip_install "wandb==0.16.6" --no-deps
pip_install appdirs sentry-sdk docker-pycreds setproctitle GitPython

# pre-seed LIBERO's one-time interactive config prompt
python - "$REPO_ROOT" <<'PY'
import os, sys, yaml

repo_root = sys.argv[1]
libero_root = os.path.join(repo_root, "src/LIBERO/libero/libero")
cfg_dir = os.path.expanduser("~/.libero")
os.makedirs(cfg_dir, exist_ok=True)
cfg_file = os.path.join(cfg_dir, "config.yaml")

if not os.path.exists(cfg_file):
    cfg = {
        "benchmark_root": libero_root,
        "bddl_files": os.path.join(libero_root, "bddl_files"),
        "init_states": os.path.join(libero_root, "init_files"),
        "datasets": os.path.join(libero_root, "../datasets"),
        "assets": os.path.join(libero_root, "assets"),
    }
    with open(cfg_file, "w") as f:
        yaml.dump(cfg, f)
PY

echo
echo "Done. Activate with: conda activate $ENV_NAME"
