"""Application configuration loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ccf_key_detector.app import RunnerConfig


@dataclass(frozen=True)
class AppConfig:
    runner: RunnerConfig


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration data from disk or return defaults."""

    if path is not None:
        raise NotImplementedError("Config file parsing will be added in a later release")
    return AppConfig(runner=RunnerConfig())
