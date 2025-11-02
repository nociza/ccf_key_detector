import struct
from typing import List, Tuple, Union

import numpy as np

from ccf_key_detector.rt_viz import (
    VisualizationConfig,
    VisualizationFrame,
    VisualizationMode,
    create_visualizer,
)


def _frame(pdf: np.ndarray, low_conf: bool = False) -> VisualizationFrame:
    total = float(np.sum(pdf))
    if total > 0.0:
        distribution = pdf / total
    else:
        distribution = np.full(pdf.shape, 1.0 / pdf.size, dtype=float)
    return VisualizationFrame(
        pdf=pdf.tolist(),
        low_confidence=low_conf,
        diagnostics={
            "rms": float(np.mean(pdf)),
            "spectral_flatness": 0.1,
            "ske_score": 0.5,
            "processing_latency_ms": 10.0,
            "ske_distribution": distribution.tolist(),
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


def test_osc_visualizer_transmits_packet(monkeypatch) -> None:
    sent_packets: list[tuple[bytes, tuple[str, int]]] = []
    sockets: list["DummySocket"] = []

    class DummySocket:
        def __init__(self, *args, **kwargs):
            self.closed = False
            sockets.append(self)

        def setblocking(self, flag: bool) -> None:
            self.flag = flag

        def sendto(self, data: bytes, endpoint: Tuple[str, int]) -> int:
            sent_packets.append((data, endpoint))
            return len(data)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("ccf_key_detector.rt_viz.osc.socket.socket", DummySocket)

    config = VisualizationConfig(
        mode=VisualizationMode.OSC_CCF,
        render=False,
        osc_host="localhost",
        osc_port=9001,
        osc_address="/tonal/pdf",
    )
    visualizer = create_visualizer(config)

    pdf = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    visualizer.push(7, _frame(pdf))

    assert sent_packets, "Expected an OSC packet to be emitted"
    payload, endpoint = sent_packets[-1]
    assert endpoint == ("localhost", 9001)

    address, tags, values = _decode_osc(payload)
    assert address == "/tonal/pdf"
    assert tags == ",iffff"
    assert values[0] == 7
    np.testing.assert_allclose(values[1:], pdf, rtol=1e-6, atol=1e-6)

    visualizer.close()
    assert sockets and sockets[0].closed


def test_osc_ske_visualizer_uses_distribution(monkeypatch) -> None:
    sent_packets: list[tuple[bytes, tuple[str, int]]] = []

    class DummySocket:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def setblocking(self, flag: bool) -> None:
            pass

        def sendto(self, data: bytes, endpoint: Tuple[str, int]) -> int:
            sent_packets.append((data, endpoint))
            return len(data)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("ccf_key_detector.rt_viz.osc.socket.socket", DummySocket)

    config = VisualizationConfig(
        mode=VisualizationMode.OSC_SKE_DIST,
        render=False,
        osc_host="127.0.0.1",
        osc_port=9010,
        osc_address="/tonal/ske",
    )
    visualizer = create_visualizer(config)

    pdf = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    frame = _frame(pdf)
    visualizer.push(3, frame)

    assert sent_packets, "Expected an OSC packet to be emitted"
    payload, endpoint = sent_packets[-1]
    assert endpoint == ("127.0.0.1", 9010)

    address, tags, values = _decode_osc(payload)
    assert address == "/tonal/ske"
    assert tags == ",iffff"
    assert values[0] == 3
    normalized = frame.diagnostics["ske_distribution"]
    np.testing.assert_allclose(values[1:], normalized, rtol=1e-6, atol=1e-6)

    visualizer.close()


def _decode_osc(packet: bytes) -> Tuple[str, str, List[Union[float, int]]]:
    def _read_string(offset: int) -> Tuple[str, int]:
        end = packet.index(b"\x00", offset)
        value = packet[offset:end].decode("utf-8")
        padded = (end + 4) & ~0x03
        return value, padded

    cursor = 0
    address, cursor = _read_string(cursor)
    tags, cursor = _read_string(cursor)
    values: List[Union[float, int]] = []
    for tag in tags[1:]:
        if tag == "i":
            (value,) = struct.unpack_from(">i", packet, cursor)
            cursor += 4
            values.append(int(value))
        elif tag == "f":
            (value,) = struct.unpack_from(">f", packet, cursor)
            cursor += 4
            values.append(float(value))
        else:  # pragma: no cover - defensive for unsupported tags
            raise AssertionError(f"Unexpected OSC type tag {tag!r}")
    return address, tags, values
