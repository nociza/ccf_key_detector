import wave
from pathlib import Path

import numpy as np

from ccf_key_detector.audio_io import AudioStreamConfig, create_audio_stream

SAMPLE_RATE = 48_000


def _write_wav(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    scaled = np.clip(data, -1.0, 0.999969482421875)
    scaled = (scaled * 32768.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())


def test_file_audio_stream_generates_overlapping_frames(tmp_path: Path) -> None:
    duration = 1.0
    samples = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    _write_wav(tmp_path / "linear.wav", samples)

    config = AudioStreamConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=0.5,
        hop_length_sec=0.25,
        use_live_input=False,
        file_path=str(tmp_path / "linear.wav"),
    )
    stream = create_audio_stream(config)
    frames = list(stream.frames())

    assert len(frames) == 3
    for index, (frame_idx, frame) in enumerate(frames):
        assert frame_idx == index
        np_frame = np.frombuffer(frame, dtype=np.float32)
        assert np_frame.size == int(config.frame_length_sec * SAMPLE_RATE)


def test_file_audio_stream_resamples(tmp_path: Path) -> None:
    src_rate = 24_000
    duration = 1.0
    t = np.arange(int(src_rate * duration), dtype=np.float64) / src_rate
    waveform = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    _write_wav(tmp_path / "tone.wav", waveform, sample_rate=src_rate)

    config = AudioStreamConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=0.5,
        hop_length_sec=0.5,
        use_live_input=False,
        file_path=str(tmp_path / "tone.wav"),
    )
    stream = create_audio_stream(config)
    frame_idx, frame = next(iter(stream.frames()))
    assert frame_idx == 0
    resampled = np.frombuffer(frame, dtype=np.float32)
    assert resampled.size == int(config.frame_length_sec * SAMPLE_RATE)
