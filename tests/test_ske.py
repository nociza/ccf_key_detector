import math

import numpy as np
import pytest

from ccf_key_detector.ske import (
    SKEBackend,
    SKEConfig,
    create_ske,
    expand_profile,
    reduce_cycle,
)
from ccf_key_detector.ske.kk_major import KK_MAJOR_PROFILE


def test_reduce_cycle_handles_non_divisible_bins() -> None:
    source_bins = 50
    pdf = np.zeros(source_bins)
    pdf[5] = 0.5
    pdf[17] = 0.3
    pdf[42] = 0.2
    reduced = reduce_cycle(pdf, reference_bins=12)
    assert math.isclose(float(reduced.sum()), float(pdf.sum()), rel_tol=1e-9)
    assert reduced.size == 12


def test_reduce_cycle_respects_circular_boundary() -> None:
    pdf = np.zeros(120)
    pdf[0] = 0.5
    pdf[-1] = 0.5
    reduced = reduce_cycle(pdf, reference_bins=12)
    assert reduced[0] > 0.0
    assert reduced[-1] > 0.0


def test_kk_major_cosine_similarity_peaks_for_in_key_profile() -> None:
    ske = create_ske(SKEConfig(mode=SKEBackend.KK_MAJOR))
    profile = np.array(KK_MAJOR_PROFILE, dtype=np.float64)
    in_key_pdf = profile / profile.sum()
    off_key_pdf = np.roll(in_key_pdf, 3)
    uniform_pdf = np.full(12, 1.0 / 12.0)

    in_score = ske.evaluate(in_key_pdf)
    off_score = ske.evaluate(off_key_pdf)
    uniform_score = ske.evaluate(uniform_pdf)

    assert in_score > off_score
    assert 0.0 <= in_score <= 1.0
    assert uniform_score < 0.3


def test_ske_zero_vector_returns_zero() -> None:
    ske = create_ske(SKEConfig(mode=SKEBackend.KK_MAJOR))
    zero_pdf = np.zeros(120)
    assert ske.evaluate(zero_pdf) == 0.0


def test_expand_profile_matches_reference_length() -> None:
    expanded = expand_profile(KK_MAJOR_PROFILE, target_bins=360)
    assert expanded.size == 360
    assert expanded.max() > 0
