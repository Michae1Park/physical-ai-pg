# physical-ai-pg

Running [OpenVLA](https://github.com/openvla/openvla) against the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation benchmark (MuJoCo, via robosuite) to reproduce the results shown on [openvla.github.io](https://openvla.github.io/).

Rendering runs headlessly via MuJoCo's EGL backend (GPU-accelerated, no display needed) — rollouts are saved as MP4 files rather than shown in an on-screen viewer, which is the practical option on a remote/SSH'd-into GPU box.

`src/openvla` and `src/LIBERO` are vendored directly in this repo (neither is an actively maintained PyPI package, and OpenVLA's LIBERO eval script only exists as source, not as an installable module) — cloning this repo is all you need to get the code. What's left is building the Python environment.

## Prerequisites

- NVIDIA GPU + driver (tested on an L40S, CUDA 12.1 toolchain via PyTorch's own wheels — no system CUDA toolkit required)
- `conda` or `mamba`
- **Python 3.10** — OpenVLA pins `tensorflow==2.15.0`, which has no Python 3.12 wheel, so a 3.10 env is required (see below)
- ~20GB free disk per LIBERO checkpoint you download, plus a few GB for the LIBERO/robosuite assets

## Installation

```bash
bash scripts/setup_env.sh
```

Creates the `openvla` conda env (Python 3.10) and installs everything: PyTorch, OpenVLA,
LIBERO, and a chain of version pins needed to make robosuite/MuJoCo/TensorFlow coexist
without segfaulting. The two required patches to OpenVLA's eval code (skip flash-attn,
stop TensorFlow from touching the GPU) are already applied in the vendored `src/openvla`.
See the comments in `scripts/setup_env.sh` for the why on each step. Safe to re-run.

This machine shows intermittent crashes unrelated to this setup — seen in pip installs,
MuJoCo's model compiler, and even plain CPython stdlib, which points to a hardware/memory
reliability issue rather than anything fixable here. If something crashes with no clear
cause, just retry; it has consistently succeeded on the next attempt.

## Quick Start

Run a small smoke test on LIBERO-Spatial (1 trial per task, 10 episodes total) using the
OpenVLA-7B checkpoint fine-tuned for that suite. `MUJOCO_GL=egl` is what makes rendering
headless/GPU-accelerated instead of requiring a display:

```bash
conda activate openvla
cd src/openvla

MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python experiments/robot/libero/run_libero_eval.py \
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
