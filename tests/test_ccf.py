import math
from typing import Tuple

import numpy as np
import pytest

from ccf_key_detector.features_ccf import CCFConfig, create_extractor


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
