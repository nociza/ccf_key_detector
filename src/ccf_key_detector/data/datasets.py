"""Dataset utilities for feature-level training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ccf_key_detector.features import (
    CCFConfig,
    build_torus,
    center_field,
    compute_cicv,
    create_extractor,
    fold_cicv,
)


@dataclass(frozen=True)
class FeatureDatasetConfig:
    sample_rate: int = 48_000
    frame_length_sec: float = 10.0
    hop_length_sec: float = 1.0
    ccf: CCFConfig = CCFConfig()
    normalize_audio: bool = True
    include_torus: bool = False


class VocalFeatureDataset(Dataset):
    """Iterate over sliding windows of audio and emit feature tensors."""

    def __init__(
        self,
        root: str | Path,
        config: FeatureDatasetConfig | None = None,
        extensions: Sequence[str] = (".wav", ".flac"),
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.config = config or FeatureDatasetConfig()
        self.extensions = tuple(ext.lower() for ext in extensions)

        self._files = self._discover_files()
        if not self._files:
            raise RuntimeError(f"No audio files found under {self.root}")

        self._index: List[Tuple[int, int, int]] = []  # (file_idx, start_frame, num_frames)
        self._build_index()
        self._extractor = create_extractor(self.config.ccf)

    def _discover_files(self) -> List[Path]:
        files = []
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() in self.extensions:
                files.append(path)
        return files

    def _build_index(self) -> None:
        self._index.clear()
        for idx, path in enumerate(self._files):
            info = sf.info(str(path))
            frame_len = int(round(self.config.frame_length_sec * info.samplerate))
            hop_len = int(round(self.config.hop_length_sec * info.samplerate))
            if frame_len <= 0 or hop_len <= 0:
                continue
            if info.frames == 0:
                continue
            start = 0
            appended = False
            while start + frame_len <= info.frames:
                self._index.append((idx, start, frame_len))
                appended = True
                start += hop_len
            if not appended:
                # Include at least one padded frame for short clips.
                self._index.append((idx, 0, info.frames))
            elif start < info.frames:
                # Capture the final partial window with padding.
                self._index.append((idx, info.frames - frame_len, frame_len))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._index)

    def __getitem__(self, idx: int):  # type: ignore[override]
        file_idx, start, frame_len = self._index[idx]
        path = self._files[file_idx]
        samples, src_rate = sf.read(str(path), start=start, stop=start + frame_len, dtype="float32")
        samples = self._ensure_mono(samples)
        if self.config.normalize_audio and np.max(np.abs(samples)) > 0:
            samples = samples / np.max(np.abs(samples))
        frame = self._resample(samples, src_rate, self.config.sample_rate)
        frame = self._pad_or_trim(frame, int(round(self.config.frame_length_sec * self.config.sample_rate)))

        pdf, diagnostics = self._extractor.compute(frame, self.config.sample_rate)
        pdf_np = np.asarray(pdf, dtype=np.float64)
        cicv = compute_cicv(pdf_np)
        folded_cicv = fold_cicv(cicv)
        center, mu, rho = center_field(pdf_np)

        sample = {
            "ccf": torch.from_numpy(pdf_np.astype(np.float32)),
            "cicv": torch.from_numpy(cicv.astype(np.float32)),
            "cicv_folded": torch.from_numpy(folded_cicv.astype(np.float32)),
            "center_field": torch.from_numpy(center.astype(np.float32)),
            "mu": torch.tensor(float(mu), dtype=torch.float32),
            "rho": torch.tensor(float(rho), dtype=torch.float32),
            "metadata": {
                "path": str(path),
                "start": start,
                "sample_rate": self.config.sample_rate,
            },
            "diagnostics": diagnostics,
        }

        if self.config.include_torus:
            torus = build_torus(pdf_np)
            sample["torus"] = torch.from_numpy(torus.astype(np.float32))
        return sample

    @staticmethod
    def _ensure_mono(samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            return samples
        return samples.mean(axis=1)

    @staticmethod
    def _pad_or_trim(frame: np.ndarray, target_len: int) -> np.ndarray:
        if frame.size == target_len:
            return frame
        if frame.size > target_len:
            return frame[:target_len]
        return np.pad(frame, (0, target_len - frame.size))

    @staticmethod
    def _resample(samples: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
        if src_rate == target_rate:
            return samples
        duration = samples.size / src_rate
        target_len = int(round(duration * target_rate))
        if target_len <= 1:
            return np.zeros(1, dtype=np.float32)
        src_times = np.linspace(0.0, duration, num=samples.size, endpoint=False)
        dst_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
        resampled = np.interp(dst_times, src_times, samples)
        return resampled.astype(np.float32)


__all__ = ["FeatureDatasetConfig", "VocalFeatureDataset"]
