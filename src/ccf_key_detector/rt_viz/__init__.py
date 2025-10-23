"""Realtime visualization utilities for tonal-center PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from .cylinder import CylinderVisualizer
from .line import LineVisualizer
from .osc import OscStreamVisualizer
from .polar import PolarVisualizer
from .surface import SurfaceVisualizer


class VisualizationMode(str, Enum):
    LINEAR = "linear"
    POLAR_SINGLE = "polar_single"
    CYLINDER_WINDOW = "cylinder_window"
    SURFACE = "surface"
    OSC_CCF = "osc_ccf"
    OSC_SKE_DIST = "osc_ske_dist"


@dataclass(frozen=True)
class VisualizationConfig:
    mode: VisualizationMode = VisualizationMode.LINEAR
    window_hops: int = 30
    fps: int = 30
    render: bool = True
    tail_length: int = 3
    normalize_ske_dist: bool = False
    osc_host: str = "127.0.0.1"
    osc_port: int = 7400
    osc_address: str = "/ccf/pdf"
    osc_include_frame_index: bool = True
    osc_max_packet_size: int = 65507


@dataclass
class VisualizationFrame:
    pdf: list[float]
    low_confidence: bool
    diagnostics: dict[str, object]


class Visualizer(Protocol):
    config: VisualizationConfig

    def push(self, frame_index: int, frame: VisualizationFrame) -> None:
        """Update the realtime view with the latest tonal-center PDF."""

    def close(self) -> None:
        """Release UI resources."""


_VISUALIZER_MAP = {
    VisualizationMode.LINEAR: LineVisualizer,
    VisualizationMode.POLAR_SINGLE: PolarVisualizer,
    VisualizationMode.CYLINDER_WINDOW: CylinderVisualizer,
    VisualizationMode.SURFACE: SurfaceVisualizer,
    VisualizationMode.OSC_CCF: lambda config: OscStreamVisualizer(config, payload="ccf"),
    VisualizationMode.OSC_SKE_DIST: lambda config: OscStreamVisualizer(config, payload="ske"),
}


def create_visualizer(config: VisualizationConfig) -> Visualizer:
    """Factory for realtime visualizers."""

    visualizer_cls = _VISUALIZER_MAP.get(config.mode)
    if visualizer_cls is None:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported visualization mode: {config.mode}")
    return visualizer_cls(config)


__all__ = [
    "VisualizationMode",
    "VisualizationConfig",
    "VisualizationFrame",
    "Visualizer",
    "create_visualizer",
]
