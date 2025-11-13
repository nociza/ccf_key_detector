import tempfile
from pathlib import Path

import numpy as np
import torch

from ccf_key_detector.models.cicv_vae import CICVVAE, CICVVAEConfig
from ccf_key_detector.ske import SKEBackend, SKEConfig, create_ske


def test_cicv_vae_forward_shapes() -> None:
    config = CICVVAEConfig(input_dim=12, hidden_dims=(16,), latent_dim=8)
    model = CICVVAE(config)
    x = torch.rand(4, 12)
    recon, mu, logvar, z = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, config.latent_dim)
    assert logvar.shape == (4, config.latent_dim)
    assert z.shape == (4, config.latent_dim)


def test_cicv_vae_ske_backend(tmp_path: Path) -> None:
    config = CICVVAEConfig(input_dim=8, hidden_dims=(8,), latent_dim=4)
    model = CICVVAE(config)
    checkpoint = {
        "model": model.state_dict(),
        "config": config.__dict__,
    }
    ckpt_path = tmp_path / "cicv_vae.pt"
    torch.save(checkpoint, ckpt_path)

    ske_config = SKEConfig(
        mode=SKEBackend.CICV_VAE,
        cicv_vae_checkpoint=str(ckpt_path),
        cicv_score_temperature=10.0,
    )
    evaluator = create_ske(ske_config)

    pdf = np.ones(8) / 8
    score = evaluator.evaluate(pdf)
    assert 0.0 < score <= 1.0
    diag = evaluator.diagnostics()
    assert "cicv_vae_loss" in diag
