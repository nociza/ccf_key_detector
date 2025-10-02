"""Audio input abstractions for realtime or file-based operation."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class AudioStreamConfig:
    """Configuration for a mono audio stream."""

    sample_rate: int
    frame_length_sec: float
    hop_length_sec: float
    use_live_input: bool
    file_path: Optional[str] = None
    input_device: Optional[int | str] = None
    block_duration_sec: Optional[float] = None

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_length_sec <= 0:
            raise ValueError("frame_length_sec must be positive")
        if self.hop_length_sec <= 0:
            raise ValueError("hop_length_sec must be positive")
        if self.frame_length_sec < self.hop_length_sec:
            raise ValueError("frame_length_sec must be >= hop_length_sec")
        if not self.use_live_input and not self.file_path:
            raise ValueError("file_path required when not using live input")
        if self.block_duration_sec is not None and self.block_duration_sec <= 0.0:
            raise ValueError("block_duration_sec must be positive when provided")


class AudioStream(Protocol):
    """Protocol for objects that yield successive mono frames."""

    config: AudioStreamConfig

    def frames(self) -> Iterable[Tuple[int, memoryview]]:
        """Yield tuples of (frame_index, mono_frame_samples)."""


class _FileAudioStream:
    """Audio stream backed by a file on disk."""

    def __init__(self, config: AudioStreamConfig) -> None:
        self.config = config
        assert self.config.file_path is not None
        data, src_rate = _load_audio_file(Path(self.config.file_path))
        if src_rate != self.config.sample_rate:
            data = _resample_linear(data, src_rate, self.config.sample_rate)
        self.samples = data

    def frames(self) -> Iterator[Tuple[int, memoryview]]:
        frame_samples = int(round(self.config.frame_length_sec * self.config.sample_rate))
        hop_samples = int(round(self.config.hop_length_sec * self.config.sample_rate))
        if frame_samples <= 0:
            raise ValueError("frame length too small for given sample rate")
        if hop_samples <= 0:
            raise ValueError("hop length too small for given sample rate")

        if len(self.samples) < frame_samples:
            padded = np.pad(self.samples, (0, frame_samples - len(self.samples)))
        else:
            padded = self.samples

        total_samples = len(padded)
        frame_index = 0
        for start in range(0, total_samples - frame_samples + 1, hop_samples):
            window = padded[start : start + frame_samples]
            yield frame_index, memoryview(window.astype(np.float32, copy=False))
            frame_index += 1


class _LiveAudioStream:
    """Audio stream that captures live audio using `sounddevice`/PortAudio."""

    def __init__(self, config: AudioStreamConfig) -> None:  # pragma: no cover - hardware specific
        self.config = config
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
        self._status_queue: queue.Queue[str] = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._last_status: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def last_status(self) -> Optional[str]:
        return self._last_status

    def frames(self) -> Iterator[Tuple[int, memoryview]]:  # pragma: no cover - hardware specific
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "sounddevice is required for live input; install the optional dependency"
            ) from exc

        frame_samples = int(round(self.config.frame_length_sec * self.config.sample_rate))
        hop_samples = int(round(self.config.hop_length_sec * self.config.sample_rate))
        if frame_samples <= 0 or hop_samples <= 0:
            raise ValueError("Invalid framing configuration for live stream")

        block_samples = hop_samples
        if self.config.block_duration_sec is not None:
            block_samples = int(round(self.config.block_duration_sec * self.config.sample_rate))
        if block_samples <= 0:
            block_samples = hop_samples

        buffer = np.zeros(frame_samples, dtype=np.float32)
        write_pos = 0
        filled = 0
        samples_since_emit = 0
        frame_index = 0

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                message = str(status)
                self._last_status = message
                try:
                    self._status_queue.put_nowait(message)
                except queue.Full:
                    pass
            # Copy mono channel data to avoid referencing the input buffer.
            block = indata[:, 0].copy()
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                # Drop the block if the consumer is too slow; record status.
                overflow = "queue_overflow"
                self._last_status = overflow
                try:
                    self._status_queue.put_nowait(overflow)
                except queue.Full:
                    pass

        stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            device=self.config.input_device,
            blocksize=block_samples,
            callback=callback,
        )

        with stream:
            while not self._stop_event.is_set():
                try:
                    chunk = self._queue.get(timeout=self.config.hop_length_sec * 2)
                except queue.Empty:
                    continue
                if chunk is None:
                    break

                if chunk.ndim != 1:
                    chunk = np.reshape(chunk, -1)
                chunk_len = int(chunk.shape[0])
                offset = 0
                while offset < chunk_len:
                    copy_len = min(chunk_len - offset, frame_samples - write_pos)
                    buffer[write_pos : write_pos + copy_len] = chunk[offset : offset + copy_len]
                    write_pos = (write_pos + copy_len) % frame_samples
                    filled = min(frame_samples, filled + copy_len)
                    samples_since_emit += copy_len
                    offset += copy_len

                    while filled == frame_samples and samples_since_emit >= hop_samples:
                        # Oldest sample sits at write_pos.
                        if write_pos == 0:
                            frame = buffer.copy()
                        else:
                            frame = np.concatenate((buffer[write_pos:], buffer[:write_pos]))
                        samples_since_emit -= hop_samples
                        if not self._status_queue.empty():
                            self._last_status = self._status_queue.get_nowait()
                        mv = memoryview(frame.astype(np.float32, copy=False))
                        yield frame_index, mv
                        frame_index += 1

        self._stop_event.clear()


def _load_audio_file(path: Path) -> Tuple[np.ndarray, int]:
    """Load a mono float32 waveform from a PCM WAV file."""

    if not path.exists():
        raise FileNotFoundError(path)

    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        pcm_data = wf.readframes(frame_count)

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 3:
        raise ValueError("24-bit WAV files are not yet supported")
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    ints = np.frombuffer(pcm_data, dtype=dtype).copy()
    if channels > 1:
        ints = ints.reshape(-1, channels).mean(axis=1)

    if dtype == np.int16:
        norm = 32768.0
    else:  # int32
        norm = 2147483648.0

    audio = ints.astype(np.float32) / norm
    return audio, sample_rate


def _resample_linear(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample using simple linear interpolation."""

    if src_rate == dst_rate:
        return data.astype(np.float32, copy=False)

    duration = len(data) / float(src_rate)
    dst_length = int(round(duration * dst_rate))
    if dst_length <= 1:
        return np.zeros(1, dtype=np.float32)

    src_times = np.linspace(0.0, duration, num=len(data), endpoint=False)
    dst_times = np.linspace(0.0, duration, num=dst_length, endpoint=False)
    resampled = np.interp(dst_times, src_times, data)
    return resampled.astype(np.float32)


def create_audio_stream(config: AudioStreamConfig) -> AudioStream:
    """Factory placeholder for the audio stream implementation."""

    if config.use_live_input:
        return _LiveAudioStream(config)
    return _FileAudioStream(config)


__all__ = [
    "AudioStreamConfig",
    "AudioStream",
    "create_audio_stream",
]
