"""Environment-driven path configuration for IKOD.

Override any directory via environment variables:
    export IKOD_CKPT_DIR=/path/to/checkpoints
    export IKOD_DATA_DIR=/path/to/benchmarks
    export IKOD_RESULTS_DIR=./results
    export HF_HOME=~/.cache/huggingface
"""
import os

def _p(name: str, default: str) -> str:
    return os.environ.get(name, default)

CKPT_DIR    = _p("IKOD_CKPT_DIR", "./checkpoints")
DATA_DIR    = _p("IKOD_DATA_DIR", "./data")
RESULTS_DIR = _p("IKOD_RESULTS_DIR", "./results")
HF_HUB      = _p("IKOD_HF_HUB", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))

__all__ = ["CKPT_DIR", "DATA_DIR", "RESULTS_DIR", "HF_HUB"]
