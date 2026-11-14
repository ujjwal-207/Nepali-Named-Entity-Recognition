"""
Reproducibility utilities for Research-nner.
Ensures deterministic behavior across all training and evaluation scripts.
Follows IEEE Reproducibility Guidelines (2024) and Pineau et al. (2021) ML Reproducibility Checklist.
"""

import random
import numpy as np
import torch
import os
import sys
import platform
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.
    Includes Python, NumPy, PyTorch (CPU + CUDA), and CUDA backend settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"✅ Seed set to {seed} | cudnn.deterministic=True | cudnn.benchmark=False")


def get_environment_info() -> dict:
    """Collect comprehensive environment information for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version,
    }

    try:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "None"
        if torch.cuda.is_available():
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass

    try:
        import datasets
        info["datasets_version"] = datasets.__version__
    except Exception:
        pass

    try:
        import seqeval
        info["seqeval_version"] = seqeval.__version__ if hasattr(seqeval, '__version__') else "unknown"
    except Exception:
        pass

    return info


def get_pip_freeze() -> dict:
    """Get all installed package versions."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, check=True)
        packages = {}
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                name, version = line.split("==", 1)
                packages[name] = version
        return packages
    except Exception:
        return {"error": "Could not run pip freeze"}


def log_environment(output_dir: str, seed: int = 42, extra_info: Optional[dict] = None) -> str:
    """Log full environment information to a JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    env_log = {
        "reproducibility": {
            "seed": seed,
            "deterministic_cudnn": True,
            "cudnn_benchmark": False,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "environment": get_environment_info(),
        "all_packages": get_pip_freeze(),
    }

    if extra_info:
        env_log["experiment_info"] = extra_info

    log_file = output_path / "environment.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(env_log, f, indent=2, default=str)

    print(f"📝 Environment logged to: {log_file}")
    return str(log_file)


def reproducibility_block(seed: int = 42, output_dir: Optional[str] = None,
                          experiment_name: str = "research-nner") -> None:
    """Full reproducibility setup. Call at start of every script."""
    set_seed(seed)

    if output_dir:
        extra = {"experiment_name": experiment_name}
        log_environment(output_dir, seed=seed, extra_info=extra)

    print(f"{'='*60}")
    print(f"  Experiment: {experiment_name} | Seed: {seed}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*60}")
