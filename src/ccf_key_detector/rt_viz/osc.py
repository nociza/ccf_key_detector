from __future__ import annotations

import socket
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from . import VisualizationConfig, VisualizationFrame


def _pad4(data: bytes) -> bytes:
    """Pad OSC string segments to a 4-byte boundary."""

    padding = (4 - (len(data) % 4)) % 4
    if padding:
        data += b"\x00" * padding
    return data


def _encode_osc_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    # OSC strings are null-terminated before padding.
    encoded += b"\x00"
    return _pad4(encoded)


class OscStreamVisualizer:
    """Visualizer that streams tonal PDFs over OSC for Max integration."""

    def __init__(self, config: "VisualizationConfig", payload: str = "ccf") -> None:
        self.config = config
        self._payload = payload
        self._socket: socket.socket | None = None

    def push(self, frame_index: int, frame: "VisualizationFrame") -> None:
        if self._payload == "ske":
            distribution = frame.diagnostics.get("ske_distribution")
            if distribution is None:
                raise ValueError("SKE distribution not available in diagnostics")
            pdf = np.asarray(distribution, dtype=np.float32)
        else:
            pdf = np.asarray(frame.pdf, dtype=np.float32)
        if pdf.ndim != 1:
            raise ValueError("OscStreamVisualizer expects 1-D pdf vectors")
        if not np.all(np.isfinite(pdf)):
            raise ValueError("Probability vector contains non-finite values")
        if self._socket is None:
            self._socket = self._create_socket()
        message = self._build_message(frame_index, pdf)
        if len(message) > self.config.osc_max_packet_size:
            raise ValueError(
                f"OSC payload ({len(message)} bytes) exceeds max packet size "
                f"{self.config.osc_max_packet_size}"
            )
        try:
            assert self._socket is not None
            self._socket.sendto(message, (self.config.osc_host, self.config.osc_port))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to send OSC packet to {self.config.osc_host}:{self.config.osc_port}"
            ) from exc

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def _create_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        return sock

    def _build_message(self, frame_index: int, pdf: np.ndarray) -> bytes:
        address = self.config.osc_address
        if not address.startswith("/"):
            address = "/" + address
        address_segment = _encode_osc_string(address)

        type_tags = ","
        payload_segments: list[bytes] = []
        if self.config.osc_include_frame_index:
            type_tags += "i"
            payload_segments.append(struct.pack(">i", int(frame_index)))

        if pdf.size:
            type_tags += "f" * pdf.size
            floats = pdf.astype(np.float32, copy=False)
            payload_segments.append(struct.pack(f">{floats.size}f", *map(float, floats)))

        type_tag_segment = _encode_osc_string(type_tags)
        return b"".join([address_segment, type_tag_segment, *payload_segments])
