# physical-ai-pg

Running [OpenVLA](https://github.com/openvla/openvla) against the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation benchmark (MuJoCo, via robosuite) to reproduce the results shown on [openvla.github.io](https://openvla.github.io/).

Rendering runs headlessly via MuJoCo's EGL backend (GPU-accelerated, no display needed) — rollouts are saved as MP4 files rather than shown in an on-screen viewer, which is the practical option on a remote/SSH'd-into GPU box.

## Prerequisites

- NVIDIA GPU + driver (tested on an L40S, CUDA 12.1 toolchain via PyTorch's own wheels — no system CUDA toolkit required)
- `conda` or `mamba`
- **Python 3.10** — OpenVLA pins `tensorflow==2.15.0`, which has no Python 3.12 wheel, so a 3.10 env is required (see below)
- ~20GB free disk per LIBERO checkpoint you download, plus a few GB for the LIBERO/robosuite assets

## Installation

```bash
# 1. Create the env
conda create -n openvla python=3.10 -y
conda activate openvla

# 2. Downgrade pip — pip >= 24 has a resolver bug that crashes when resolving
#    some of the older pinned packages below (tensorflow==2.15, old wandb).
pip install "pip==23.3.2"

# 3. PyTorch (match to your CUDA; this matches driver CUDA 12.x/13.x via forward compat)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

# 4. Clone and install OpenVLA
mkdir -p src && cd src
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
cd ../..

# 5. Clone LIBERO
cd src
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .          # this "succeeds" but produces a broken editable install — see step 6
cd ../..

# 6. Fix LIBERO's editable install
# LIBERO's package layout is LIBERO/libero/libero/ (an implicit namespace package `libero`
# wrapping the real package `libero.libero`). setuptools' modern editable-install finder
# can't resolve that layout, so `import libero` fails outside the LIBERO directory. Work
# around it with a plain .pth file pointing at the repo root instead:
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/src/LIBERO" > "$SITE/zz_libero_root.pth"

# 7. Install LIBERO's extra runtime deps
cd src/openvla
pip install -r experiments/robot/libero/libero_requirements.txt
pip install "numpy<2"
cd ../..

# 8. Pin mujoco — robosuite==1.4.1 (pulled in by LIBERO) declares mujoco>=2.3.0 with no
# upper bound, so pip installs the latest by default. Anything >=3.x segfaults/asserts
# (`get_joint_qpos_addr` AssertionError) when robosuite creates an environment.
pip install "mujoco==2.3.7"

# 9. Fix the protobuf / tensorflow-metadata / wandb version chain
# tensorflow==2.15.0 needs an older protobuf than the default tensorflow-metadata pulls in.
pip install "tensorflow-metadata==1.14.0"
# That downgrades protobuf to 3.20.3, which breaks a modern `wandb`. Pin an old wandb instead
# (only used for optional logging; --no-deps avoids re-triggering the pip resolver bug):
pip install "wandb==0.16.6" --no-deps
pip install appdirs sentry-sdk docker-pycreds setproctitle GitPython
```

Two small source patches are also required in `src/openvla/experiments/robot/openvla_utils.py`:

1. **Skip flash-attn** (avoids needing a system CUDA toolkit / `nvcc` to compile it — fine for
   a quick eval, no measurable quality loss):
   ```python
   # attn_implementation="flash_attention_2",
   attn_implementation="sdpa",
   ```

2. **Stop TensorFlow from touching the GPU.** TensorFlow is only used for the RLDS data-loading
   utilities (unused at eval time), but importing it unconditionally grabs a CUDA/EGL context
   that collides with MuJoCo's own EGL context and segfaults the process the moment an
   environment is created. Add this right after `import tensorflow as tf`:
   ```python
   import tensorflow as tf
   tf.config.set_visible_devices([], "GPU")
   ```

Finally, pre-seed LIBERO's one-time interactive config prompt so it doesn't block a
non-interactive run:

```bash
python -c "
import os, yaml
libero_root = 'src/LIBERO/libero/libero'
cfg_dir = os.path.expanduser('~/.libero')
os.makedirs(cfg_dir, exist_ok=True)
cfg = {
  'benchmark_root': libero_root,
  'bddl_files': os.path.join(libero_root, 'bddl_files'),
  'init_states': os.path.join(libero_root, 'init_files'),
  'datasets': os.path.join(libero_root, '../datasets'),
  'assets': os.path.join(libero_root, 'assets'),
}
with open(os.path.join(cfg_dir, 'config.yaml'), 'w') as f:
    yaml.dump(cfg, f)
"
```

## Quick Start

Run a small smoke test on LIBERO-Spatial (1 trial per task, 10 episodes total) using the
OpenVLA-7B checkpoint fine-tuned for that suite. `MUJOCO_GL=egl` is what makes rendering
headless/GPU-accelerated instead of requiring a display:

```bash
conda activate openvla
cd src/openvla

MUJOCO_GL=egl python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --use_wandb False
```

The first run downloads the ~15GB checkpoint from Hugging Face. Rollout videos are written to
`src/openvla/rollouts/<date>/`, one MP4 per episode, named with the task and whether it
succeeded. A smoke test run (2026-08-26) got 8/10 (80%) success — in line with the paper's
84.7% average for LIBERO-Spatial (measured over 500 trials).

### Other benchmark suites

Swap `--task_suite_name` and `--pretrained_checkpoint` to run the other three LIBERO suites:

| Suite | `--task_suite_name` | `--pretrained_checkpoint` |
|---|---|---|
| Spatial | `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` |
| Object | `libero_object` | `openvla/openvla-7b-finetuned-libero-object` |
| Goal | `libero_goal` | `openvla/openvla-7b-finetuned-libero-goal` |
| Long (LIBERO-10) | `libero_10` | `openvla/openvla-7b-finetuned-libero-10` |

Bump `--num_trials_per_task` up to 50 to reproduce the paper's full evaluation protocol (500
episodes per suite — expect several hours per suite on a single GPU).
