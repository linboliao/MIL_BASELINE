"""Reproducibility helpers for MIL training — ADDITIVE, opt-in via env var.

Nothing here changes default behaviour. `set_global_seed()` in
utils/general_utils.py already seeds python/numpy/torch/cuda and sets
cudnn.deterministic=True / benchmark=False. That is enough for *same
machine, same library stack* run-to-run stability of the forward pass,
but NOT enough for:
  * cross-environment reproducibility (different torch / cuDNN / GPU arch),
  * the backward pass (some CUDA reductions use atomics -> nondeterministic
    unless torch.use_deterministic_algorithms(True)),
  * DataLoader workers' own python/numpy RNG (no worker_init_fn today),
  * python hash randomisation (PYTHONHASHSEED).

Turn the stronger path on with:  export MIL_DETERMINISM=1
(optionally MIL_DETERMINISM=strict to make nondeterministic ops raise
instead of warn).

IMPORTANT: cross-environment *bitwise* identity is still not guaranteed
when torch / cuDNN / cuBLAS / GPU architecture differ. The point of this
module is (a) to remove *within-environment* run-to-run noise so a
multi-seed study is meaningful, and (b) to record the full environment
so any residual difference is attributable.
"""
import hashlib
import json
import os
import platform
import subprocess
import random

import numpy as np
import torch


def _flagged(name="MIL_DETERMINISM"):
    return os.environ.get(name, "").strip().lower()


def deterministic_requested():
    return _flagged() in ("1", "true", "yes", "on", "strict")


def enable_full_determinism(seed: int):
    """Superset of set_global_seed(). Call INSTEAD of set_global_seed when
    MIL_DETERMINISM is set; otherwise this is never invoked and behaviour
    is unchanged."""
    mode = _flagged()
    # PYTHONHASHSEED must be set by the launcher before interpreter startup.
    # cuBLAS needs this for a deterministic GEMM workspace (>= CUDA 10.2)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # kill TF32 (Ampere+) — otherwise matmul path depends on GPU arch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    warn_only = (mode != "strict")
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:                      # torch < 1.11 has no warn_only
        torch.use_deterministic_algorithms(True)
    print(f"enable_full_determinism: seed={seed} mode={mode or 'on'} "
          f"warn_only={warn_only}")


def seed_worker(worker_id):
    """DataLoader worker_init_fn — seed each worker's numpy / random from
    torch's per-worker seed so augmentation / sampling inside __getitem__
    (there is none today, but future-proof) is reproducible."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ------------------------------------------------------------------ metadata

def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_commit(cwd):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None



def collect_run_metadata(args=None, split_csv=None, repo_dir=None):
    import sys
    from datetime import datetime, timezone
    repo_dir = repo_dir or os.getcwd()
    def git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=repo_dir,
                                       stderr=subprocess.DEVNULL)
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        diff = git("diff", "HEAD", "--binary")
        status = git("status", "--porcelain").decode()
    except (OSError, subprocess.CalledProcessError):
        commit, diff, status = None, b"", None
    config = dict(args) if isinstance(args, dict) else None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python_executable": sys.executable,
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "git_commit": commit,
        "git_status": status, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "split_csv": str(split_csv) if split_csv else None,
        "split_csv_sha256": _sha256_file(split_csv) if split_csv else None,
        "config": config,
        "determinism_requested": deterministic_requested(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

def write_run_metadata(out_dir, args=None, split_csv=None, repo_dir=None):
    from pathlib import Path
    md = collect_run_metadata(args, split_csv, repo_dir)
    p = Path(out_dir)/"run_metadata.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(md, f, indent=2, ensure_ascii=False, allow_nan=False)
    return str(p)
