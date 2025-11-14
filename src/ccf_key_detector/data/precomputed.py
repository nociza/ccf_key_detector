"""Utilities for offline precomputation of tonal features."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ccf_key_detector.features import (
    CCFConfig,
    CCFExtractor,
    CenterFieldConfig,
    build_torus,
    center_field,
    compute_cicv,
    create_extractor,
    fold_cicv,
)


@dataclass(frozen=True)
class PrecomputeConfig:
    """Configuration for feature precomputation."""

    sample_rate: int = 48_000
    frame_length_sec: float = 10.0
    hop_length_sec: float = 1.0
    ccf: CCFConfig = CCFConfig()
    center: CenterFieldConfig = CenterFieldConfig()
    include_torus: bool = True
    include_folded_cicv: bool = True
    store_pdf: bool = True

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["ccf"] = self.ccf.to_dict()
        payload["center"] = self.center.to_dict()
        return payload

    def cfg_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class FrameFeatures:
    pdf: np.ndarray
    cicv: np.ndarray
    center: np.ndarray
    mu: float
    rho: float
    torus: Optional[np.ndarray] = None
    cicv_folded: Optional[np.ndarray] = None
    diagnostics: Optional[Dict[str, float]] = None


def compute_features(
    frame: np.ndarray,
    config: PrecomputeConfig,
    *,
    extractor: Optional[CCFExtractor] = None,
) -> FrameFeatures:
    ccf_extractor = extractor or create_extractor(config.ccf)
    pdf, diagnostics = ccf_extractor.compute(memoryview(frame), config.sample_rate)
    pdf_arr = np.asarray(pdf, dtype=np.float64)
    cicv = compute_cicv(pdf_arr)
    center_map, mu, rho = center_field(pdf_arr, config.center)

    cicv_folded = fold_cicv(cicv) if config.include_folded_cicv else None
    torus = build_torus(pdf_arr) if config.include_torus else None
    diag_dict = dict(diagnostics)

    return FrameFeatures(
        pdf=np.asarray(pdf_arr, dtype=np.float32),
        cicv=cicv.astype(np.float32),
        center=center_map.astype(np.float32),
        mu=float(mu),
        rho=float(rho),
        torus=None if torus is None else torus.astype(np.float32),
        cicv_folded=None if cicv_folded is None else cicv_folded.astype(np.float32),
        diagnostics=diag_dict,
    )


def center_field_with_config(pdf: np.ndarray, config: PrecomputeConfig) -> FrameFeatures:
    return compute_features(pdf, config)


@dataclass(frozen=True)
class PrecomputedEntry:
    audio: Path
    features: Path
    n_frames: int
    starts: Sequence[int]


@dataclass(frozen=True)
class PrecomputedManifest:
    cfg_id: str
    config: Dict[str, object]
    audio_root: Path
    entries: Tuple[PrecomputedEntry, ...]

    @classmethod
    def load(cls, path: Path) -> "PrecomputedManifest":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        entries = tuple(
            PrecomputedEntry(
                audio=Path(entry["audio"]),
                features=Path(entry["features"]),
                n_frames=int(entry["n_frames"]),
                starts=tuple(entry.get("starts", [])),
            )
            for entry in payload["entries"]
        )
        return cls(
            cfg_id=payload["cfg_id"],
            config=payload["config"],
            audio_root=Path(payload["audio_root"]),
            entries=entries,
        )


class PrecomputedFeatureDataset(torch.utils.data.Dataset):
    """Dataset backed by offline precomputed feature archives."""

    def __init__(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest_path = manifest_path.resolve()
        self.manifest = PrecomputedManifest.load(self.manifest_path)
        self._feature_root = self.manifest_path.parent
        self._cached_entry_idx: Optional[int] = None
        self._cached_arrays: Optional[Dict[str, np.ndarray]] = None
        self._index: List[Tuple[int, int]] = []
        for entry_idx, entry in enumerate(self.manifest.entries):
            for frame_idx in range(entry.n_frames):
                self._index.append((entry_idx, frame_idx))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._index)

    def __getitem__(self, idx: int):  # type: ignore[override]
        entry_idx, frame_idx = self._index[idx]
        entry = self.manifest.entries[entry_idx]
        arrays = self._get_entry_arrays(entry_idx, entry)

        sample = {
            "ccf": torch.from_numpy(arrays["ccf"][frame_idx]).float(),
            "cicv": torch.from_numpy(arrays["cicv"][frame_idx]).float(),
            "center_field": torch.from_numpy(arrays["center"][frame_idx]).float(),
            "mu": torch.tensor(float(arrays["mu"][frame_idx]), dtype=torch.float32),
            "rho": torch.tensor(float(arrays["rho"][frame_idx]), dtype=torch.float32),
            "metadata": {
                "audio": str(entry.audio),
                "features": str(entry.features),
                "frame_index": frame_idx,
                "start": int(arrays["starts"][frame_idx]),
                "cfg_id": self.manifest.cfg_id,
            },
        }

        if "cicv_folded" in arrays:
            sample["cicv_folded"] = torch.from_numpy(arrays["cicv_folded"][frame_idx]).float()
        if "torus" in arrays:
            sample["torus"] = torch.from_numpy(arrays["torus"][frame_idx]).float()

        diag = {
            key: float(arrays[f"diag_{key}"][frame_idx])
            for key in ["rms", "spectral_flatness", "raw_energy", "fft_size", "n_bins"]
            if f"diag_{key}" in arrays
        }
        sample["diagnostics"] = diag

        return sample

    def _get_entry_arrays(self, entry_idx: int, entry: PrecomputedEntry) -> Dict[str, np.ndarray]:
        if self._cached_entry_idx == entry_idx and self._cached_arrays is not None:
            return self._cached_arrays
        arrays = self._load_entry(entry)
        self._cached_entry_idx = entry_idx
        self._cached_arrays = arrays
        return arrays

    def _load_entry(self, entry: PrecomputedEntry) -> Dict[str, np.ndarray]:
        path = (self._feature_root / entry.features).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
# fmt: off
        data = np.load(path, allow_pickle=False)
        try:
            arrays = {key: data[key] for key in data.files}
        finally:
            data.close()
# fmt: on
        return arrays


__all__ = [
    "PrecomputeConfig",
    "FrameFeatures",
    "compute_features",
    "PrecomputedManifest",
    "PrecomputedFeatureDataset",
]
