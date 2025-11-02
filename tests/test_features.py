import math
from typing import Tuple

import numpy as np
import pytest

from ccf_key_detector.features import (
    CCFConfig,
    CenterFieldConfig,
    build_torus,
    center_field,
    column_sums,
    compute_cicv,
    create_extractor,
    fold_cicv,
    rotate_pdf,
    rotate_torus,
)


SAMPLE_RATE = 48_000
FRAME_DURATION = 0.5  # seconds


def _tone(freq_hz: float, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * FRAME_DURATION), dtype=np.float64) / SAMPLE_RATE
    waveform = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    return waveform.astype(np.float32)


def _mixed_tones(freqs: Tuple[float, ...], amplitude: float = 0.35) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * FRAME_DURATION), dtype=np.float64) / SAMPLE_RATE
    waveform = np.zeros_like(t)
    for freq in freqs:
        waveform += amplitude * np.sin(2.0 * np.pi * freq * t)
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform.astype(np.float32)


def test_pdf_is_normalised() -> None:
    config = CCFConfig(n_bins=180, smoothing_sigma_bins=1.0)
    extractor = create_extractor(config)

    frame = _tone(330.0)
    pdf, diagnostics = extractor.compute(memoryview(frame), SAMPLE_RATE)

    assert math.isclose(sum(pdf), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    expected_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    assert diagnostics["rms"] == pytest.approx(expected_rms, rel=1e-6)
    assert diagnostics["spectral_flatness"] >= 0.0
    assert diagnostics["spectral_flatness"] <= 1.0


def test_single_tone_aligns_with_expected_bin() -> None:
    n_bins = 240
    config = CCFConfig(n_bins=n_bins, smoothing_sigma_bins=0.0)
    extractor = create_extractor(config)

    freq_hz = 220.0 * 2 ** 0.25
    frame = _tone(freq_hz)

    pdf, _ = extractor.compute(memoryview(frame), SAMPLE_RATE)
    peak_index = int(np.argmax(pdf))

    expected_fraction = math.fmod(math.log2(freq_hz / config.f_ref_hz), 1.0)
    expected_fraction = expected_fraction if expected_fraction >= 0 else expected_fraction + 1.0
    expected_index = int(round(expected_fraction * n_bins)) % n_bins

    distance = min((peak_index - expected_index) % n_bins, (expected_index - peak_index) % n_bins)
    assert distance <= 1


def test_octave_equivalence() -> None:
    config = CCFConfig(n_bins=180, smoothing_sigma_bins=1.5)
    extractor = create_extractor(config)

    base_pdf, _ = extractor.compute(memoryview(_tone(275.0)), SAMPLE_RATE)
    octave_pdf, _ = extractor.compute(memoryview(_tone(550.0)), SAMPLE_RATE)

    base_arr = np.array(base_pdf)
    octave_arr = np.array(octave_pdf)
    assert int(base_arr.argmax()) == int(octave_arr.argmax())
    assert np.sum(np.abs(base_arr - octave_arr)) < 0.2


def test_zero_energy_returns_uniform_pdf() -> None:
    config = CCFConfig(n_bins=100, smoothing_sigma_bins=0.5)
    extractor = create_extractor(config)

    frame = np.zeros(int(SAMPLE_RATE * FRAME_DURATION), dtype=np.float32)
    pdf, _ = extractor.compute(memoryview(frame), SAMPLE_RATE)

    assert np.allclose(pdf, np.full(config.n_bins, 1.0 / config.n_bins))


def test_noise_has_higher_spectral_flatness_than_tone() -> None:
    config = CCFConfig(n_bins=120, smoothing_sigma_bins=0.0)
    extractor = create_extractor(config)

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.5, int(SAMPLE_RATE * FRAME_DURATION)).astype(np.float32)
    tone = _tone(220.0)

    _, noise_diag = extractor.compute(memoryview(noise), SAMPLE_RATE)
    _, tone_diag = extractor.compute(memoryview(tone), SAMPLE_RATE)

    assert noise_diag["spectral_flatness"] > tone_diag["spectral_flatness"]


def test_smoothing_preserves_total_probability_mass() -> None:
    config = CCFConfig(n_bins=90, smoothing_sigma_bins=3.0)
    extractor = create_extractor(config)

    frame = _tone(300.0)
    pdf, _ = extractor.compute(memoryview(frame), SAMPLE_RATE)

    assert math.isclose(sum(pdf), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert np.count_nonzero(np.array(pdf) > 1e-4) > 2


def test_mixed_tones_exhibit_multiple_peaks() -> None:
    freqs = (250.0, 300.0, 440.0)
    config = CCFConfig(n_bins=180, smoothing_sigma_bins=1.0)
    extractor = create_extractor(config)

    frame = _mixed_tones(freqs)
    pdf, _ = extractor.compute(memoryview(frame), SAMPLE_RATE)
    arr = np.array(pdf)
    baseline = np.median(arr)
    threshold = max(baseline * 5, baseline + 1e-6)

    n_bins = config.n_bins
    for freq in freqs:
        cycle = math.fmod(math.log2(freq / config.f_ref_hz), 1.0)
        cycle = cycle if cycle >= 0 else cycle + 1.0
        center = int(round(cycle * n_bins)) % n_bins
        window_indices = [(center + offset) % n_bins for offset in (-1, 0, 1)]
        window_peak = arr[window_indices].max()
        assert window_peak > threshold, f"Peak near {freq} Hz below expected prominence"


def test_cicv_rotation_invariant() -> None:
    config = CCFConfig(n_bins=120, smoothing_sigma_bins=0.5)
    extractor = create_extractor(config)
    pdf, _ = extractor.compute(memoryview(_tone(330.0)), SAMPLE_RATE)
    directed = compute_cicv(np.array(pdf))
    rotated_pdf = rotate_pdf(np.array(pdf), shift=5)
    rotated_directed = compute_cicv(rotated_pdf)
    np.testing.assert_allclose(directed, rotated_directed, rtol=1e-9, atol=1e-9)


def test_cicv_transposition_equivalence_for_triads() -> None:
    n_bins = 36
    major = np.zeros(n_bins)
    major[[0, 4, 7]] = 1.0
    major /= major.sum()
    transposed_major = rotate_pdf(major, shift=7)
    augmented = np.zeros(n_bins)
    augmented[[0, 4, 8]] = 1.0
    augmented /= augmented.sum()

    g_major = compute_cicv(major)
    g_transposed = compute_cicv(transposed_major)
    g_aug = compute_cicv(augmented)

    np.testing.assert_allclose(g_major, g_transposed, rtol=1e-9, atol=1e-9)
    assert np.linalg.norm(g_major - g_aug) > 0.05


def test_folded_cicv_normalization_and_symmetry() -> None:
    rng = np.random.default_rng(0)
    pdf = rng.random(45)
    pdf /= pdf.sum()
    directed = compute_cicv(pdf)
    folded = fold_cicv(directed)

    assert math.isclose(directed.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(folded.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9)

    # Symmetry check: directed[k] and directed[-k] contribute equally.
    for k in range(1, directed.size // 2):
        np.testing.assert_allclose(directed[k], directed[-k], rtol=1e-9, atol=1e-9)


def test_torus_column_sums_match_cicv() -> None:
    rng = np.random.default_rng(1)
    pdf = rng.random(32)
    pdf /= pdf.sum()
    torus = build_torus(pdf)
    directed = compute_cicv(pdf)
    np.testing.assert_allclose(column_sums(torus), directed, rtol=1e-9, atol=1e-9)


def test_torus_rotates_with_pdf() -> None:
    rng = np.random.default_rng(2)
    pdf = rng.random(20)
    pdf /= pdf.sum()
    torus = build_torus(pdf)
    rotated_pdf = rotate_pdf(pdf, shift=3)
    rotated_torus = build_torus(rotated_pdf)
    np.testing.assert_allclose(rotated_torus, rotate_torus(torus, shift=3), rtol=1e-9, atol=1e-9)


def test_center_field_single_peak_high_concentration() -> None:
    n_bins = 120
    pdf = np.zeros(n_bins)
    pdf[0] = 1.0
    center, mu, rho = center_field(pdf)
    assert center.sum() > 0
    assert min(abs(mu), 1 - abs(mu)) < 1e-6
    assert rho > 0.98


def test_center_field_two_centers_reduce_concentration() -> None:
    n_bins = 120
    single = np.zeros(n_bins)
    single[0] = 1.0
    double = np.zeros(n_bins)
    double[0] = double[n_bins // 2] = 0.5
    _, _, rho_single = center_field(single)
    _, _, rho_double = center_field(double)
    assert rho_double < rho_single
    assert rho_double < 0.8


def test_center_field_rotates_with_pdf() -> None:
    n_bins = 120
    pdf = np.zeros(n_bins)
    pdf[10] = 1.0
    center, mu, rho = center_field(pdf)
    rotated_pdf = rotate_pdf(pdf, shift=7)
    _, rotated_mu, rotated_rho = center_field(rotated_pdf)
    expected_mu = (mu - (7 / n_bins)) % 1.0
    assert math.isclose(rotated_rho, rho, rel_tol=1e-9, abs_tol=1e-9)
    diff = min(abs(rotated_mu - expected_mu), 1.0 - abs(rotated_mu - expected_mu))
    assert diff < 1e-6
