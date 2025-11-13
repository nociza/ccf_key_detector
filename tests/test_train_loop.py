from pathlib import Path

import json
import numpy as np
import soundfile as sf

from ccf_key_detector.data import FeatureDatasetConfig, VocalFeatureDataset
from ccf_key_detector.features import CCFConfig
from ccf_key_detector.models import SmallVAEConfig
from ccf_key_detector.train.train_small import TrainingConfig, train_small

SAMPLE_RATE = 48_000


def _write_tone(path: Path, freq: float = 220.0, duration: float = 0.3) -> None:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
    waveform = 0.5 * np.sin(2.0 * np.pi * freq * t)
    sf.write(path, waveform.astype(np.float32), SAMPLE_RATE)


def test_train_small_runs(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone(audio_path)

    feature_cfg = FeatureDatasetConfig(
        sample_rate=SAMPLE_RATE,
        frame_length_sec=0.2,
        hop_length_sec=0.2,
        ccf=CCFConfig(n_bins=40),
    )
    training_cfg = TrainingConfig(
        dataset_root=tmp_path,
        feature=feature_cfg,
        model=SmallVAEConfig(input_channels=1, input_height=1, input_width=40, latent_dim=6),
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        max_steps_per_epoch=1,
        device="cpu",
    )

    artifacts = train_small(training_cfg, return_state=True, record_history=True)
    assert artifacts is not None
    assert artifacts.history is not None
    assert len(artifacts.history) == 1

    fine_tune_cfg = TrainingConfig(
        dataset_root=tmp_path,
        feature=feature_cfg,
        model=training_cfg.model,
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        max_steps_per_epoch=1,
        device="cpu",
    )
    train_small(fine_tune_cfg, state=artifacts.state)

    dataset = VocalFeatureDataset(tmp_path, config=feature_cfg)
    sample = dataset[0]
    npz_path = tmp_path / "tone_features.npz"
    arrays = {
        "ccf": sample["ccf"].unsqueeze(0).numpy(),
        "cicv": sample["cicv"].unsqueeze(0).numpy(),
        "center": sample["center_field"].unsqueeze(0).numpy(),
        "cicv_folded": sample["cicv_folded"].unsqueeze(0).numpy(),
        "mu": np.array([float(sample["mu"])], dtype=np.float32),
        "rho": np.array([float(sample["rho"])], dtype=np.float32),
        "starts": np.array([sample["metadata"]["start"]], dtype=np.int64),
    }
    np.savez(npz_path, **arrays)

    manifest = {
        "cfg_id": "test_cfg",
        "config": feature_cfg.ccf.to_dict(),
        "audio_root": str(tmp_path),
        "entries": [
            {
                "audio": "tone.wav",
                "features": npz_path.name,
                "n_frames": 1,
                "starts": [sample["metadata"]["start"]],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    precomputed_cfg = TrainingConfig(
        dataset_root=tmp_path,
        feature=feature_cfg,
        model=training_cfg.model,
        batch_size=1,
        num_epochs=1,
        lr=1e-3,
        max_steps_per_epoch=1,
        device="cpu",
        precomputed_manifest=manifest_path,
    )
    train_small(precomputed_cfg)
