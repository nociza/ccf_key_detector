tldr: 

STEP 0: get the dependencies sorted out
STEP 1: run
python scripts/run_live_ccf.py --viz-mode osc_ccf
or
python scripts/run_live_ccf.py --viz-mode osc_ske_dist

STEP 2: Listen at 7400.

STEP 2 alternative: respectively open 'ccf_osc_monitor.scd' or 'ske_dist_osc_monitor.scd' and evaluate. Change according to usage.

For any question, contact Eva at hanyinliang.eva@berkeley.edu.

# Realtime Tonal-Center Probability (Continuous Chroma)

This project implements a realtime tonal-center estimator built on a continuous chroma representation anchored at 220 Hz. Audio is ingested as overlapping mono frames, pre-filtered, projected onto the unit log-frequency cycle, evaluated against single-key profiles, and optionally displayed in several visualization modes.

## Highlights

- Continuous chroma feature extractor with configurable smoothing and diagnostics
- KK-major single-key evaluator plus circular scan to obtain a tonal distribution
- File and live-audio streaming back ends with lightweight resampling
- Realtime visualizers (line, polar, cylinder, surface, OSC stream) for rapid feedback
- Convenience scripts for generating synthetic material and validating SKE behaviour

## Repository Layout

```
src/ccf_key_detector/
  audio_io/        # Input abstraction, WAV loading, linear resampling
  prefilter/       # Deterministic filters (bypass/high-pass)
  features_ccf/    # Continuous chroma feature extractor
  transpose/       # Circular shift operator and tonal scan helpers
  ske/             # Single-key evaluator implementations
  rt_viz/          # Realtime visualization back ends
  app/             # Runner that stitches the full pipeline together
config/            # Runtime configuration dataclasses
scripts/           # CLI utilities and demo scripts
tests/             # Pytest-based regression and behaviour checks
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Python 3.10 or newer is required. The realtime pipeline relies on PortAudio via `sounddevice`; install the appropriate system packages if you plan to use live capture. Visualization modes that open a window depend on `pyqtgraph`, `PyQt5`, and `PyOpenGL`.

## Usage

- Generate a KK-major-profile waveform and report its SKE score:
  ```bash
  PYTHONPATH=src python scripts/generate_kk_profile_audio.py
  ```
- Produce a three-tone test clip and plot its continuous chroma PDF:
  ```bash
  PYTHONPATH=src python scripts/generate_mixed_tones.py \
      --output-wav build/mixed_tones.wav \
      --output-plot build/mixed_tones_ccf.png
  ```
- Analyse an existing WAV file and visualise its tonal distribution:
  ```bash
  PYTHONPATH=src python scripts/plot_ske_distribution.py path/to/audio.wav
  ```
- Run the realtime pipeline (live input by default, or `--audio-file` for playback):
  ```bash
  PYTHONPATH=src python scripts/run_live_ccf.py --viz-mode polar_single
  ```

Generated artefacts land in `build/` and are excluded from version control.

### Command Reference

**`generate_kk_profile_audio.py`**
- `--duration`: Length of the synthesized clip in seconds (default `5.0`).
- `--sample-rate`: Sample rate for synthesis and analysis (default `48000`).
- `--output`: Target WAV path (default `build/kk_profile.wav`).

**`generate_mixed_tones.py`**
- `--output-wav`: Destination WAV path for the synthetic mixture.
- `--output-plot`: Destination PNG path for the continuous chroma plot.
- `--duration`: Clip length in seconds.
- `--sample-rate`: Output sample rate (resampled if needed).
- `--amplitude`: Peak amplitude per tone (before clipping).
- `--n-bins`: Number of CCF bins used for analysis.
- `--smoothing`: Circular Gaussian smoothing sigma in bins.
- `--frame-length`: Length of the analysis frame (seconds); zero-padded if the clip is shorter.

**`plot_ske_distribution.py`**
- `audio_path`: Input audio file (WAV recommended).
- `--output`: Destination PNG for the combined plot.
- `--n-bins`: Number of continuous chroma bins.
- `--smoothing`: Smoothing sigma in bins.
- `--frame-length`: Analysis window length (seconds).
- `--sample-rate`: Target sample rate (linear resampling if the source differs).

**`run_live_ccf.py`**
- `--audio-file`: Optional WAV file for offline playback; omit to use a live input device.
- `--sample-rate`: Processing sample rate (Hz).
- `--frame-length`: Analysis frame length (seconds).
- `--hop-length`: Hop size (seconds).
- `--n-bins`: Continuous chroma resolution.
- `--viz-mode`: Visualization backend (`linear`, `polar_single`, `cylinder_window`, `surface`, `osc_ccf`, `osc_ske_dist`).
- `--window-hops`: History length for windowed visualizers.
- `--tail-length`: Tail length for polar visualization.
- `--device`: PortAudio device identifier (index or string) when capturing live audio.
- `--headless`: Skip rendering windows (useful with OSC modes).
- `--normalize-ske-dist`: Emit normalised SKE probabilities instead of raw scores.
- `--print-stats`: Log diagnostics (RMS, spectral flatness, SKE peaks) for each hop.
- `--osc-host`: Destination host for OSC streaming (default `127.0.0.1`).
- `--osc-port`: UDP port for OSC streaming (default `7400`).
- `--osc-address`: OSC address pattern for transmitted packets (default `/ccf/pdf`; a leading slash is added automatically if omitted).
- `--osc-skip-frame-index`: Omit the hop index from OSC payloads.
- `--osc-max-packet-size`: Maximum packet size in bytes before raising an error (default `65507`).

### OSC Integration

Select the OSC pipeline by setting `--viz-mode osc_ccf` (continuous chroma PDF bins) or `--viz-mode osc_ske_dist` (tonal distribution from the SKE scan). Each hop emits a packet with address `/ccf/pdf` (customisable via `--osc-address`) containing:

1. Optional integer hop index (disabled with `--osc-skip-frame-index`).
2. One float per chroma bin representing either raw SKE scores or the normalised distribution (depending on `--normalize-ske-dist`).

Configure the destination host and port using `--osc-host` and `--osc-port`. Keep `--headless` enabled when no on-screen visualization is needed.

#### Payload Format

- OSC address: `/ccf/pdf`
- Type tags: `,iffff…` (comma, optional integer, then floats per bin)
- Payload layout after the address:

  | Slot | Type            | Meaning                                                        |
  |------|-----------------|----------------------------------------------------------------|
  | 0    | *optional* int  | Hop index (`frame_index`) when `osc_include_frame_index` is `True` |
  |0 or 1| float           | First value (CCF bin or SKE likelihood)                        |
  | …    | float           | Remaining values ordered by bin index                          |

Example with 180 bins and hop index enabled:

```
/ccf/pdf ,ifffff...  42  0.0048 0.0037 0.0021 … 0.0059
```

When `--osc-skip-frame-index` is supplied, the integer slot is omitted and the payload is pure floats. The message length equals `n_bins + (1 if frame index present else 0)`. In `osc_ske_dist` mode the floats correspond to the SKE scores stored in `diagnostics["ske_distribution"]` (normalised if `--normalize-ske-dist` is enabled).

#### SuperCollider Monitors

- `scripts/ccf_osc_monitor.scd`: listens for `osc_ccf` output and renders the continuous chroma PDF.
- `scripts/ske_dist_osc_monitor.scd`: listens for `osc_ske_dist` output and plots the tonal distribution likelihoods.

Load the relevant file in SuperCollider and evaluate it to start listening on port `7400`. If you change `--osc-port`, update the script accordingly.

For remote consumers, point `--osc-host` at the target machine (e.g., `--osc-host 192.168.1.42`). High-resolution bin counts increase packet size, so ensure your OSC client accepts datagrams up to `--osc-max-packet-size`.

## Testing

```bash
PYTHONPATH=src pytest
```

Tests cover the CCF extractor, SKE evaluator, transposition logic, and selected visualization utilities.
