import torch

from ccf_key_detector.models import (
    SmallVAE,
    SmallVAEConfig,
    center_to_polar,
    split_center_texture,
)


def test_small_vae_forward_1d() -> None:
    config = SmallVAEConfig(input_channels=1, input_height=1, input_width=180, latent_dim=8)
    model = SmallVAE(config)
    x = torch.randn(4, 1, 1, 180)
    recon, mu, logvar, z = model(x)
    assert recon.shape == x.shape
    assert mu.shape == (4, config.latent_dim)
    assert logvar.shape == (4, config.latent_dim)
    assert z.shape == (4, config.latent_dim)


def test_small_vae_forward_2d() -> None:
    config = SmallVAEConfig(input_channels=1, input_height=32, input_width=64, latent_dim=10)
    model = SmallVAE(config)
    x = torch.randn(2, 1, 32, 64)
    recon, mu, logvar, _ = model(x)
    assert recon.shape == x.shape
    assert mu.shape[-1] == config.latent_dim
    assert logvar.shape[-1] == config.latent_dim


def test_split_center_texture_shapes() -> None:
    latent = torch.randn(5, 12)
    center, texture = split_center_texture(latent, center_dim=2)
    assert center.shape == (5, 2)
    assert texture.shape == (5, 10)


def test_split_center_texture_raises_for_small_latent() -> None:
    latent = torch.randn(3, 1)
    try:
        split_center_texture(latent, center_dim=2)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for insufficient latent dims")


def test_center_to_polar_conversion() -> None:
    latent = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    radius, angle = center_to_polar(latent)
    torch.testing.assert_close(radius, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(angle, torch.tensor([0.0, 0.25], dtype=angle.dtype))
