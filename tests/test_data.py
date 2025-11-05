import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ccf_key_detector.data import (
    FeatureDatasetConfig,
    PrecomputedFeatureDataset,
    PrecomputeConfig,
    VocalFeatureDataset,
)
from ccf_key_detector.features import CCFConfig

SAMPLE_RATE = 48_000


def _write_tone(path: Path, freq: float = 330.0, duration: float = 0.5) -> None:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
    waveform = 0.5 * np.sin(2.0 * np.pi * freq * t)
    sf.write(path, waveform.astype(np.float32), SAMPLE_RATE)


def test_dataset_single_window(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)

    config = FeatureDatasetConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=0.25,
        hop_length_sec=0.25,
        ccf=CCFConfig(n_bins=60, smoothing_sigma_bins=0.0),
        include_torus=True,
    )
    dataset = VocalFeatureDataset(tmp_path, config=config)
    assert len(dataset) == 2  # two hops in 0.5 seconds with hop=0.25

    sample = dataset[0]
    assert sample["ccf"].shape == (60,)
    assert math.isclose(float(sample["ccf"].sum()), 1.0, rel_tol=1e-6)
    assert sample["cicv"].shape == (60,)
    assert sample["cicv_folded"].ndim == 1
    assert sample["center_field"].shape == (60,)
    assert sample["torus"].shape == (60, 60)
    assert isinstance(sample["metadata"], dict)
    assert "diagnostics" in sample


def test_dataset_resample(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone_low_sr.wav"
    low_sr = 16_000
    t = np.arange(int(low_sr * 0.3), dtype=np.float64) / low_sr
    waveform = 0.5 * np.sin(2.0 * np.pi * 220.0 * t)
    sf.write(audio_path, waveform.astype(np.float32), low_sr)

    config = FeatureDatasetConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=0.2,
        hop_length_sec=0.2,
        ccf=CCFConfig(n_bins=40),
    )
    dataset = VocalFeatureDataset(tmp_path, config=config)
    sample = dataset[0]
    assert sample["ccf"].shape == (40,)
    assert isinstance(sample["mu"], torch.Tensor)
    assert 0.0 <= float(sample["rho"]) <= 1.0


def test_precomputed_dataset_roundtrip(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path, freq=261.63, duration=0.25)

    feature_cfg = FeatureDatasetConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=0.25,
        hop_length_sec=0.25,
        ccf=CCFConfig(n_bins=48, smoothing_sigma_bins=0.5),
        include_torus=True,
    )
    dataset = VocalFeatureDataset(tmp_path, config=feature_cfg)
    sample = dataset[0]

    cfg = PrecomputeConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=0.25,
        hop_length_sec=0.25,
        ccf=feature_cfg.ccf,
        include_torus=True,
    )
    cfg_id = cfg.cfg_id()

    feature_root = tmp_path / f"cfg_{cfg_id}"
    feature_root.mkdir()
    npz_path = feature_root / "tone.npz"

    arrays = {
        "ccf": sample["ccf"].unsqueeze(0).numpy(),
        "cicv": sample["cicv"].unsqueeze(0).numpy(),
        "center": sample["center_field"].unsqueeze(0).numpy(),
        "mu": np.array([float(sample["mu"])], dtype=np.float32),
        "rho": np.array([float(sample["rho"])], dtype=np.float32),
        "starts": np.array([sample["metadata"]["start"]], dtype=np.int64),
        "cicv_folded": sample["cicv_folded"].unsqueeze(0).numpy(),
        "torus": sample["torus"].unsqueeze(0).numpy(),
    }
    diagnostics = sample["diagnostics"]
    for key, value in diagnostics.items():
        arrays[f"diag_{key}"] = np.array([value], dtype=np.float32)

    np.savez(npz_path, **arrays)

    manifest = {
        "cfg_id": cfg_id,
        "config": cfg.to_dict(),
        "audio_root": str(tmp_path),
        "entries": [
            {
                "audio": "tone.wav",
                "features": "tone.npz",
                "n_frames": 1,
                "starts": [sample["metadata"]["start"]],
            }
        ],
    }
    manifest_path = feature_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    precomputed = PrecomputedFeatureDataset(manifest_path)
    loaded = precomputed[0]

    assert torch.allclose(loaded["ccf"], sample["ccf"])
    assert torch.allclose(loaded["cicv"], sample["cicv"])
    assert torch.allclose(loaded["center_field"], sample["center_field"])
    assert torch.isclose(loaded["mu"], sample["mu"])
    assert torch.isclose(loaded["rho"], sample["rho"])
    assert loaded["metadata"]["audio"] == "tone.wav"
