"""Runs a LIBERO env in a genuinely isolated OS process, so on-screen GLX rendering never
shares a process with the loaded CUDA model. Mixing a resident CUDA model with on-screen GLX
rendering in one process is unstable on this box's NVIDIA driver -- reproduced as a segfault
during model loading, a hang requiring a force-quit, and (rarely) a clean run, all with
identical code.

IMPORTANT: this worker is launched via `subprocess.Popen([sys.executable, __file__, ...])`,
NOT `multiprocessing.Process`. multiprocessing's "spawn" start method looks like it gives you a
fresh interpreter, but its bootstrap (`multiprocessing.spawn._fixup_main_from_path`) actually
re-executes the *parent script* (e.g. run_libero_eval.py) as `__mp_main__` first, so it can
rebuild `sys.modules['__main__']` before unpickling the target function. run_libero_eval.py
imports libero_utils/openvla_utils/robot_utils at module scope, and those import
tensorflow/torch/transformers -- so a `multiprocessing.Process` "worker" silently re-imports the
entire CUDA/TF stack into the child anyway, defeating the whole point of isolating it. Launching
this file directly as its own `python env_worker.py` process sidesteps that: the child's
__main__ is this module, which never imports torch/tensorflow/transformers, so it never touches
CUDA at all. Communication with the parent happens over two OS pipes passed by fd number (not
stdin/stdout, which robosuite/mujoco/gym freely write warnings to).
"""

import os
import pickle
import struct
import subprocess
import sys

# Ensure `experiments` is importable when this file is exec'd directly as a script (its own
# __main__), same as run_libero_eval.py does for its own imports.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402


def _read_exact(fd, n):
    buf = b""
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            return None  # EOF: other end closed/died
        buf += chunk
    return buf


def _send_msg(fd, obj):
    data = pickle.dumps(obj)
    os.write(fd, struct.pack("!Q", len(data)))
    os.write(fd, data)


def _recv_msg(fd):
    header = _read_exact(fd, 8)
    if header is None:
        return None
    (length,) = struct.unpack("!Q", header)
    data = _read_exact(fd, length)
    if data is None:
        return None
    return pickle.loads(data)


def _build_env(task_suite_name, task_id, resolution, render_live):
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    if render_live:
        env = ControlEnv(has_renderer=True, render_camera="frontview", **env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    initial_states = task_suite.get_task_init_states(task_id)
    return env, task_description, initial_states


def _worker_main(task_suite_name, task_id, resolution, render_live, cmd_fd, result_fd):
    try:
        env, task_description, initial_states = _build_env(task_suite_name, task_id, resolution, render_live)
    except Exception as e:
        _send_msg(result_fd, ("error", repr(e)))
        return
    _send_msg(result_fd, ("ready", (task_description, initial_states)))

    while True:
        msg = _recv_msg(cmd_fd)
        if msg is None:  # parent died/closed without sending "close"
            return
        cmd, payload = msg
        try:
            if cmd == "reset":
                _send_msg(result_fd, ("ok", env.reset()))
            elif cmd == "set_init_state":
                _send_msg(result_fd, ("ok", env.set_init_state(payload)))
            elif cmd == "step":
                _send_msg(result_fd, ("ok", env.step(payload)))
            elif cmd == "render":
                _send_msg(result_fd, ("ok", env.render()))
            elif cmd == "close":
                env.close()
                _send_msg(result_fd, ("ok", None))
                return
        except Exception as e:
            _send_msg(result_fd, ("error", repr(e)))
            return


class RemoteLiberoEnv:
    """Drop-in stand-in for a LIBERO env object (reset/set_init_state/step/render/close), but
    the real env lives in a separate `python env_worker.py` subprocess. See module docstring
    for why this has to be a real subprocess and not `multiprocessing.Process`."""

    def __init__(self, task_suite_name, task_id, resolution=256, render_live=False):
        cmd_r, cmd_w = os.pipe()  # parent writes cmd_w -> child reads cmd_r
        result_r, result_w = os.pipe()  # child writes result_w -> parent reads result_r
        os.set_inheritable(cmd_r, True)
        os.set_inheritable(result_w, True)

        self._proc = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                task_suite_name,
                str(task_id),
                str(resolution),
                "1" if render_live else "0",
                str(cmd_r),
                str(result_w),
            ],
            pass_fds=(cmd_r, result_w),
        )
        os.close(cmd_r)
        os.close(result_w)
        self._cmd_fd = cmd_w
        self._result_fd = result_r

        status, payload = self._wait_for_result("startup")
        if status == "error":
            raise RuntimeError(f"LIBERO env worker failed to start: {payload}")
        self.task_description, self.initial_states = payload

    def _wait_for_result(self, cmd):
        result = _recv_msg(self._result_fd)
        if result is None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError(
                f"LIBERO env worker process died unexpectedly during {cmd!r} "
                f"(exit code {self._proc.returncode})"
            )
        return result

    def _call(self, cmd, payload=None):
        _send_msg(self._cmd_fd, (cmd, payload))
        status, result = self._wait_for_result(cmd)
        if status == "error":
            raise RuntimeError(f"LIBERO env worker error on {cmd!r}: {result}")
        return result

    def reset(self):
        return self._call("reset")

    def set_init_state(self, state):
        return self._call("set_init_state", state)

    def step(self, action):
        return self._call("step", action)

    def render(self):
        return self._call("render")

    def close(self):
        try:
            result = self._call("close")
        finally:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        return result


if __name__ == "__main__":
    _task_suite_name, _task_id, _resolution, _render_live, _cmd_fd, _result_fd = sys.argv[1:7]
    _worker_main(
        _task_suite_name,
        int(_task_id),
        int(_resolution),
        _render_live == "1",
        int(_cmd_fd),
        int(_result_fd),
    )
