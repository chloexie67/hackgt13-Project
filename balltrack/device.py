"""Choosing the compute device, and serialising GPU work across threads."""
import threading

# Apple's Metal backend crashes if two threads encode GPU work at once; every model call that
# may overlap with background calibration takes this lock.
GPU_LOCK = threading.Lock()


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
