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
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
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
    repo_dir = repo_dir or os.getcwd()
    md = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_commit": _git_commit(repo_dir),
        "determinism_env": os.environ.get("MIL_DETERMINISM", ""),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    try:
        import sklearn
        md["sklearn"] = sklearn.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            md["gpu_name"] = torch.cuda.get_device_name(0)
            md["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            md["gpu_count"] = torch.cuda.device_count()
        except Exception:
            pass
        try:
            md["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"], stderr=subprocess.DEVNULL
            ).decode().splitlines()[0].strip()
        except Exception:
            pass
    if args is not None:
        try:
            md["seed"] = int(args.General.seed)
            md["dataset_name"] = str(args.Dataset.DATASET_NAME)
            md["model_name"] = str(args.General.MODEL_NAME)
        except Exception:
            pass
    if split_csv:
        md["split_csv"] = split_csv
        md["split_csv_sha256"] = _sha256_file(split_csv)
    return md


def write_run_metadata(out_dir, args=None, split_csv=None, repo_dir=None):
    md = collect_run_metadata(args, split_csv, repo_dir)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "run_metadata.json")
    with open(p, "w") as f:
        json.dump(md, f, indent=1, ensure_ascii=False)
    return p
