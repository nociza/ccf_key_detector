import math

import numpy as np

from ccf_key_detector.rt_viz import (
    VisualizationConfig,
    VisualizationFrame,
    VisualizationMode,
    create_visualizer,
)


def _frame(pdf: np.ndarray, low_conf: bool = False) -> VisualizationFrame:
    return VisualizationFrame(
        pdf=pdf.tolist(),
        low_confidence=low_conf,
        diagnostics={
            "rms": float(np.mean(pdf)),
            "spectral_flatness": 0.1,
            "ske_score": 0.5,
            "processing_latency_ms": 10.0,
        },
    )


def test_linear_visualizer_headless_tail_buffer() -> None:
    config = VisualizationConfig(mode=VisualizationMode.LINEAR, tail_length=4, render=False)
    visualizer = create_visualizer(config)
    for idx in range(6):
        pdf = np.zeros(10)
        pdf[(idx + 2) % 10] = 1.0
        visualizer.push(idx, _frame(pdf))
    assert len(visualizer._state.tail) == config.tail_length  # type: ignore[attr-defined]
    assert visualizer._state.tail_indices[-1] == 5  # type: ignore[attr-defined]
    visualizer.close()


def test_polar_visualizer_headless_tail_buffer() -> None:
    config = VisualizationConfig(mode=VisualizationMode.POLAR_SINGLE, tail_length=4, render=False)
    visualizer = create_visualizer(config)
    for idx in range(6):
        pdf = np.zeros(12)
        pdf[(idx + 3) % 12] = 1.0
        visualizer.push(idx, _frame(pdf))
    assert len(visualizer._state.tail) == config.tail_length  # type: ignore[attr-defined]
    assert visualizer._state.tail_indices[-1] == 5  # type: ignore[attr-defined]
    visualizer.close()


def test_cylinder_visualizer_headless_ring_buffer() -> None:
    config = VisualizationConfig(mode=VisualizationMode.CYLINDER_WINDOW, window_hops=5, render=False)
    visualizer = create_visualizer(config)
    for idx in range(7):
        pdf = np.zeros(8)
        pdf[idx % 8] = 1.0
        visualizer.push(idx, _frame(pdf))
    assert len(visualizer._state.ring_buffer) == config.window_hops  # type: ignore[attr-defined]
    assert visualizer._state.indices[-1] == 6  # type: ignore[attr-defined]
    visualizer.close()


def test_surface_visualizer_headless_buffers() -> None:
    config = VisualizationConfig(mode=VisualizationMode.SURFACE, window_hops=4, render=False)
    visualizer = create_visualizer(config)
    for idx in range(6):
        pdf = np.zeros(12)
        pdf[(idx + 2) % 12] = 1.0
        diagnostics = {
            "rms": 0.1,
            "spectral_flatness": 0.2,
            "ske_distribution": (pdf / pdf.sum()).tolist(),
        }
        frame = VisualizationFrame(pdf=pdf.tolist(), low_confidence=False, diagnostics=diagnostics)
        visualizer.push(idx, frame)
    assert len(visualizer._state.ccf_buffer) == config.window_hops  # type: ignore[attr-defined]
    assert visualizer._state.indices[-1] == 5  # type: ignore[attr-defined]
    visualizer.close()
