"""Device selection helpers for PyTorch workloads."""

from __future__ import annotations

import torch


def resolve_device(preferred: str | torch.device | None = "auto") -> torch.device:
    """Return the best available torch device for a user preference.

    ``auto`` and ``gpu`` prefer CUDA, then MPS, and finally CPU. Explicit CUDA
    requests fall back to CPU when CUDA is unavailable so the project remains
    runnable on machines without a GPU.
    """
    if isinstance(preferred, torch.device):
        preferred_text = preferred.type
    else:
        preferred_text = str(preferred or "auto").lower()

    if preferred_text in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred_text.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return torch.device("cpu")

    if preferred_text == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            print("MPS requested but unavailable; falling back to CPU.", flush=True)
            return torch.device("cpu")

    return torch.device(preferred_text)


def configure_torch_backend(device: torch.device) -> None:
    """Enable backend options that are useful for this fixed-size CNN workload."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
