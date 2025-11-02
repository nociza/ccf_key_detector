import math
import wave
from pathlib import Path

import numpy as np

from ccf_key_detector.app import FrameResult, RunnerConfig, create_runner
from ccf_key_detector.features import CCFConfig
from ccf_key_detector.prefilter import PreFilterConfig

SAMPLE_RATE = 48_000


def _write_wav(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    scaled = np.clip(data, -1.0, 0.999969482421875)
    scaled = (scaled * 32768.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())


def test_runner_produces_pdf_for_tone(tmp_path: Path) -> None:
    duration = 2.0
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
    frequency = 220.0 * 2 ** (5 / 12)
    waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    audio_path = tmp_path / "tone.wav"
    _write_wav(audio_path, waveform)

    config = RunnerConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=1.0,
        hop_length_sec=0.5,
        n_bins=180,
        use_live_input=False,
        audio_file_path=str(audio_path),
        pre_filter=PreFilterConfig(enable=False),
        ccf=CCFConfig(
            n_bins=180,
            f_ref_hz=220.0,
            smoothing_sigma_bins=1.0,
            window="hann",
        ),
    )

    runner = create_runner(config)
    results = list(runner.iterate(max_hops=2))
    assert len(results) == 2
    for result in results:
        assert isinstance(result, FrameResult)
        assert math.isclose(sum(result.pdf), 1.0, rel_tol=1e-9, abs_tol=1e-9)
        assert "rms" in result.diagnostics
        assert float(result.diagnostics["rms"]) > 0.0
        assert "processing_latency_ms" in result.diagnostics
        assert float(result.diagnostics["processing_latency_ms"]) >= 0.0
        assert "ske_score" in result.diagnostics
        assert 0.0 <= float(result.diagnostics["ske_score"]) <= 1.0
        distribution = result.diagnostics.get("ske_distribution")
        assert isinstance(distribution, list)
        assert "audio_status" not in result.diagnostics
    first_peak = max(enumerate(results[0].pdf), key=lambda item: item[1])[0]
    second_peak = max(enumerate(results[1].pdf), key=lambda item: item[1])[0]
    assert first_peak == second_peak
