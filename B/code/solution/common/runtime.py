"""CPU/runtime controls shared by scalable feature builders."""

from __future__ import annotations

import os


_THREAD_LIMITER = None


def default_workers() -> int:
    """Conservative default for the target i9-9900/62GB server."""
    available = os.cpu_count() or 1
    return max(1, min(8, available))


def configure_cpu_threads(workers: int) -> None:
    """Bound native thread pools before NumPy/LightGBM-heavy work starts.

    This prevents a Python worker pool and BLAS/OpenMP from multiplying each
    other into dozens of runnable threads and duplicating large work buffers.
    Existing explicit user settings are respected.
    """
    value = str(max(1, int(workers)))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, value)
    # When called after NumPy/SciPy has already loaded its native runtime,
    # environment variables alone are too late.  scikit-learn installs
    # threadpoolctl, so use it when available and retain the controller.
    global _THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMITER = threadpool_limits(limits=int(value))
    except ImportError:
        _THREAD_LIMITER = None
