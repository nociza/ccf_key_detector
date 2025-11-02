from pathlib import Path

import numpy as np
import soundfile as sf

from ccf_key_detector.data import FeatureDatasetConfig
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

    train_small(training_cfg)
