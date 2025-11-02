import math

import numpy as np

from ccf_key_detector.features import CCFConfig, create_extractor
from ccf_key_detector.ske import SKEBackend, SKEConfig, create_ske, reduce_cycle
from ccf_key_detector.ske.kk_major import KK_MAJOR_PROFILE


SAMPLE_RATE = 48_000
DURATION = 10.0


def _synthesise_kk_profile_waveform() -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * DURATION), dtype=np.float64) / SAMPLE_RATE
    waveform = np.zeros_like(t)
    amplitudes = _compute_profile_amplitudes()
    for idx, amplitude in enumerate(amplitudes):
        freq = 220.0 * (2.0 ** (idx / 12.0))
        waveform += amplitude * np.sin(2.0 * np.pi * freq * t)
    waveform /= np.max(np.abs(waveform))
    return waveform.astype(np.float32)


def _compute_profile_amplitudes() -> np.ndarray:
    extractor = create_extractor(CCFConfig(smoothing_sigma_bins=0.0))
    basis = []
    t = np.arange(int(SAMPLE_RATE * DURATION), dtype=np.float64) / SAMPLE_RATE
    for idx in range(12):
        freq = 220.0 * (2.0 ** (idx / 12.0))
        sine = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
        pdf, _ = extractor.compute(memoryview(sine), SAMPLE_RATE)
        reduced = reduce_cycle(pdf, 12)
        basis.append(reduced)
    basis_matrix = np.stack(basis, axis=1)
    target = np.array(KK_MAJOR_PROFILE, dtype=np.float64)
    target /= target.sum()
    active = np.ones(12, dtype=bool)
    weights = np.zeros(12, dtype=np.float64)
    for _ in range(12):
        B = basis_matrix[:, active]
        A = np.vstack([B, np.ones((1, B.shape[1]), dtype=np.float64)])
        b = np.concatenate([target, [1.0]])
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        weights[active] = solution
        if np.all(weights[active] >= -1e-9):
            weights = np.clip(weights, 0.0, None)
            break
        worst = np.argmin(weights)
        active[worst] = False
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total == 0.0:
        weights = np.full_like(weights, 1.0 / weights.size)
    else:
        weights /= total
    amplitudes = np.sqrt(weights)
    return amplitudes


def test_ske_high_for_kk_profile_waveform() -> None:
    waveform = _synthesise_kk_profile_waveform()
    extractor = create_extractor(CCFConfig(smoothing_sigma_bins=0.0))
    pdf, _ = extractor.compute(memoryview(waveform), SAMPLE_RATE)

    ske = create_ske(SKEConfig(mode=SKEBackend.KK_MAJOR))
    score = ske.evaluate(pdf)

    assert score > 0.88
