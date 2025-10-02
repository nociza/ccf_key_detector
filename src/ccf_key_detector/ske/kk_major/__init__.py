"""Krumhansl–Kessler major profile evaluator helpers."""

from __future__ import annotations

import numpy as np

KK_MAJOR_PROFILE: tuple[float, ...] = (
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
)


def cosine_similarity(profile: np.ndarray | tuple[float, ...], candidate: np.ndarray) -> float:
    """Compute the cosine similarity between ``profile`` and ``candidate``.

    Both operands are treated as vectors in Euclidean space with dimension equal
    to the KK profile length (12). The function guards against zero vectors by
    returning 0.0 to avoid undefined divisions.
    """

    prof = np.asarray(profile, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if prof.shape != cand.shape:
        raise ValueError("profile and candidate must share the same shape")

    prof_norm = np.linalg.norm(prof)
    cand_norm = np.linalg.norm(cand)
    if prof_norm == 0.0 or cand_norm == 0.0:
        return 0.0
    similarity = float(np.dot(prof, cand) / (prof_norm * cand_norm))
    return max(-1.0, min(1.0, similarity))


__all__ = ["KK_MAJOR_PROFILE", "cosine_similarity"]
