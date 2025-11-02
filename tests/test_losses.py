import math

import numpy as np

from ccf_key_detector.features import build_torus, compute_cicv, rotate_pdf
from ccf_key_detector.losses import (
    apply_free_bits,
    center_loss,
    circular_distance,
    js_cicv_loss,
    kl_cicv_loss,
    l2_cicv_loss,
    kl_warmup,
    phase_loss,
)


def test_l2_cicv_loss_rotation_invariant() -> None:
    pdf = np.zeros(36)
    pdf[[0, 4, 7]] = 1.0
    pdf /= pdf.sum()
    rotated_pdf = rotate_pdf(pdf, shift=5)
    g = compute_cicv(pdf)
    g_rot = compute_cicv(rotated_pdf)
    assert l2_cicv_loss(g, g_rot) < 1e-12


def test_z_losses_positive_for_different_structures() -> None:
    pdf_major = np.zeros(36)
    pdf_major[[0, 4, 7]] = 1.0
    pdf_major /= pdf_major.sum()
    pdf_aug = np.zeros(36)
    pdf_aug[[0, 4, 8]] = 1.0
    pdf_aug /= pdf_aug.sum()
    g_major = compute_cicv(pdf_major)
    g_aug = compute_cicv(pdf_aug)
    assert l2_cicv_loss(g_major, g_aug) > 0.05
    assert kl_cicv_loss(g_major + 1e-9, g_aug + 1e-9) > 0
    assert js_cicv_loss(g_major, g_aug) > 0


def test_phase_loss_aligns_shift() -> None:
    pdf = np.zeros(24)
    pdf[[0, 3, 7]] = 1.0
    pdf /= pdf.sum()
    torus = build_torus(pdf)
    shifted = build_torus(rotate_pdf(pdf, shift=4))
    assert phase_loss(torus, shifted) < 1e-12


def test_phase_loss_with_inversion_option() -> None:
    pdf = np.zeros(24)
    pdf[[0, 3, 7]] = 1.0
    pdf /= pdf.sum()
    torus = build_torus(pdf)
    inverted_pdf = pdf[::-1]
    inverted_torus = build_torus(inverted_pdf)
    loss_without = phase_loss(torus, inverted_torus, allow_inversion=False)
    loss_with = phase_loss(torus, inverted_torus, allow_inversion=True)
    assert loss_with <= loss_without


def test_center_loss_zero_for_identical_centers() -> None:
    assert center_loss(0.2, 0.8, 0.2, 0.9) == 0.0


def test_center_loss_increases_with_distance() -> None:
    loss_close = center_loss(0.1, 0.9, 0.1 + 1 / 12, 0.9)
    loss_far = center_loss(0.1, 0.9, 0.1 + 0.5, 0.9)
    assert loss_far > loss_close


def test_circular_distance_modular() -> None:
    assert math.isclose(circular_distance(0.0, 1.0), 0.0)
    assert circular_distance(0.0, 0.5) == 2.0


def test_kl_warmup_schedule() -> None:
    assert kl_warmup(0, 100, 1.0) == 0.0
    assert math.isclose(kl_warmup(50, 100, 1.0), 0.5)
    assert kl_warmup(200, 100, 1.0) == 1.0


def test_apply_free_bits() -> None:
    kl = np.array([0.1, 0.4, 0.9])
    adjusted = apply_free_bits(kl, free_bits=0.3)
    np.testing.assert_allclose(adjusted, np.array([0.0, 0.1, 0.6]))
