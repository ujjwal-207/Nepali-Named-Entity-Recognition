"""
Reproducibility utilities for Nepali Named Entity Recognition.

Ensures deterministic execution across all training and evaluation runs,
in accordance with IEEE and ML reproducibility standards.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, PyTorch CPU, and CUDA backends.

    Args:
        seed: Integer random seed value.
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


def get_environment_info() -> Dict[str, Any]:
    """Collect system, hardware, and library environment metadata.

    Returns:
        Dictionary containing hardware specifications, OS, and package versions.
    """
    info: Dict[str, Any] = {
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
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
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
        info["seqeval_version"] = getattr(seqeval, "__version__", "unknown")
    except Exception:
        pass

    return info


def get_pip_freeze() -> Dict[str, str]:
    """Capture pinned package versions via pip freeze.

    Returns:
        Mapping of installed package names to their exact versions.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages: Dict[str, str] = {}
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                name, version = line.split("==", 1)
                packages[name] = version
        return packages
    except Exception:
        return {"error": "Could not execute pip freeze"}


def log_environment(
    output_dir: str,
    seed: int = 42,
    extra_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Log full reproducibility metadata to environment.json.

    Args:
        output_dir: Directory where the log file is saved.
        seed: Random seed applied to the run.
        extra_info: Additional metadata or experiment arguments.

    Returns:
        String path to the saved environment file.
    """
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

    return str(log_file)


def reproducibility_block(
    seed: int = 42,
    output_dir: Optional[str] = None,
    experiment_name: str = "research-nner",
) -> None:
    """Initialize reproducibility settings at the start of an experiment.

    Args:
        seed: Random seed value.
        output_dir: Optional directory to log the execution environment.
        experiment_name: Identifier for the current experiment run.
    """
    set_seed(seed)

    if output_dir:
        extra = {"experiment_name": experiment_name}
        log_environment(output_dir, seed=seed, extra_info=extra)

    print("=" * 60)
    print(f"Experiment: {experiment_name} | Seed: {seed}")
    print(f"Timestamp:  {datetime.now().isoformat()}")
    print("=" * 60)
