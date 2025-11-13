"""Realtime application runner for the tonal-center estimator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, Optional, Sequence

import numpy as np

from ccf_key_detector.audio_io import AudioStream, AudioStreamConfig, create_audio_stream
from ccf_key_detector.features import CCFConfig, create_extractor
from ccf_key_detector.prefilter import PreFilter, PreFilterConfig, create_prefilter
from ccf_key_detector.rt_viz import VisualizationConfig
from ccf_key_detector.ske import SKEConfig, create_ske
from ccf_key_detector.transpose import TranspositionConfig, create_operator, scan_distribution


@dataclass
class RunnerConfig:
    sample_rate: int = 48_000
    frame_length_sec: float = 10.0
    hop_length_sec: float = 1.0
    n_bins: int = 120
    viz_window_hops: int = 30
    logging_level: str = "warn"
    use_live_input: bool = True
    audio_file_path: Optional[str] = None
    pre_filter: PreFilterConfig = field(default_factory=PreFilterConfig)
    ccf: CCFConfig = field(default_factory=CCFConfig)
    ske: SKEConfig = field(default_factory=SKEConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    audio_input_device: Optional[int | str] = None
    audio_block_duration_sec: Optional[float] = None


@dataclass
class FrameResult:
    frame_index: int
    pdf: list[float]
    diagnostics: Dict[str, object]


class Runner:
    """High-level orchestrator for the realtime tonal-center estimator."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        audio_config = AudioStreamConfig(
            sample_rate=config.sample_rate,
            frame_length_sec=config.frame_length_sec,
            hop_length_sec=config.hop_length_sec,
            use_live_input=config.use_live_input,
            file_path=config.audio_file_path,
            input_device=config.audio_input_device,
            block_duration_sec=config.audio_block_duration_sec,
        )
        self.audio_stream: AudioStream = create_audio_stream(audio_config)
        self.prefilter: PreFilter = create_prefilter(config.pre_filter)
        ccf_config = config.ccf
        if ccf_config.n_bins != config.n_bins:
            ccf_config = replace(ccf_config, n_bins=config.n_bins)
        self.ccf_extractor = create_extractor(ccf_config)
        self.ske = create_ske(config.ske)
        self.transpose = create_operator(TranspositionConfig(n_bins=config.n_bins))
        self._closed = False

    def iterate(self, max_hops: Optional[int] = None) -> Iterator[FrameResult]:
        """Yield successive tonal-center PDFs for each audio hop."""

        count = 0
        for frame_index, frame in self.audio_stream.frames():
            start_time = time.perf_counter()
            processed = self.prefilter.process(frame, self.config.sample_rate)
            pdf, diagnostics = self.ccf_extractor.compute(processed, self.config.sample_rate)
            diagnostics = dict(diagnostics)
            diagnostics["processing_latency_ms"] = (time.perf_counter() - start_time) * 1000.0
            tonal_pdf, ske_diag = self._scan_tonal_distribution(pdf)
            diagnostics["ske_score"] = float(np.max(tonal_pdf))
            diagnostics["ske_distribution"] = tonal_pdf.tolist()
            diagnostics["ske_peak_index"] = int(np.argmax(tonal_pdf))
            diagnostics.update(ske_diag)
            last_status = getattr(self.audio_stream, "last_status", None)
            if callable(last_status):
                status_value = last_status()
                if status_value:
                    diagnostics.setdefault("audio_status", status_value)
            yield FrameResult(frame_index=frame_index, pdf=pdf, diagnostics=diagnostics)
            count += 1
            if max_hops is not None and count >= max_hops:
                break

    def run(self, max_hops: Optional[int] = None) -> None:  # pragma: no cover - orchestration
        for _ in self.iterate(max_hops=max_hops):
            pass

    def close(self) -> None:
        self._closed = True
        stop = getattr(self.audio_stream, "stop", None)
        if callable(stop):
            stop()

    def _scan_tonal_distribution(self, pdf: Sequence[float]) -> tuple[np.ndarray, dict]:
        normalized = scan_distribution(pdf, self.ske, self.transpose)
        if getattr(self.ske, "requires_transposition", True):
            if not self.config.viz.normalize_ske_dist:
                vector = np.asarray(pdf, dtype=np.float64)
                scores = np.empty(self.config.n_bins, dtype=np.float64)
                for shift in range(self.config.n_bins):
                    rotated = self.transpose.apply(vector, shift)
                    scores[shift] = self.ske.evaluate(rotated)
                return scores, getattr(self.ske, "diagnostics", lambda: {})()
            return normalized, getattr(self.ske, "diagnostics", lambda: {})()

        score = self.ske.evaluate(pdf)
        distribution = np.full(self.config.n_bins, score, dtype=np.float64)
        return distribution, getattr(self.ske, "diagnostics", lambda: {})()


def create_runner(config: RunnerConfig) -> Runner:
    """Construct the runner with instantiated pipeline components."""

    return Runner(config)


__all__ = [
    "RunnerConfig",
    "Runner",
    "create_runner",
    "FrameResult",
]
