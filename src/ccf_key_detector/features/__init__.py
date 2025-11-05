"""Feature extraction utilities for tonal-center analysis."""

from .ccf import CCFConfig, CCFExtractor, create_extractor
from .cicv import cicv_pair, compute_cicv, fold_cicv, rotate_pdf
from .centerfield import CenterFieldConfig, center_field
from .torus import build_torus, column_sums, rotate_torus

try:  # Optional dependency: torch
    from .torus import build_torus_torch  # type: ignore
except ImportError:  # pragma: no cover - exercised on CPU-only setups
    def build_torus_torch(*args, **kwargs):  # type: ignore
        raise RuntimeError("build_torus_torch requires PyTorch to be installed") from None

__all__ = [
    "CCFConfig",
    "CCFExtractor",
    "CenterFieldConfig",
    "build_torus",
    "center_field",
    "cicv_pair",
    "column_sums",
    "compute_cicv",
    "create_extractor",
    "build_torus_torch",
    "fold_cicv",
    "rotate_pdf",
    "rotate_torus",
]
