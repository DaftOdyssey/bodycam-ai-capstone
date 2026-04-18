# Bodycam AI Capstone

AI-powered bodycam system for real-time tension detection and multimodal incident analysis.

## Description

This project implements a two-stage pipeline for automated incident detection and analysis using bodycam hardware:

1. **Edge Pipeline (`edge_pipeline.py`)** -- Runs on the bodycam device. Captures live audio and video, performs real-time audio classification using YAMNet (TensorFlow Hub) to detect events such as shouting, screaming, or gunshots, and streams buffered media over ZMQ when an incident is triggered.

2. **Slow Brain Worker (`slow_brain_worker.py`)** -- Runs offline on a processing node. Receives assembled audio/video from a finalized incident, transcribes speech (Whisper / faster-whisper), scores video frames for behavioral categories (X-CLIP / CLIP / heuristic fallback), fuses multimodal signals into a risk score, and generates semantic embeddings suitable for vector search (pgvector).

3. **Test Harness (`minimal_test_harness.py`)** -- A lightweight ZMQ server that bridges the two stages during development. It accepts control and data messages from the edge pipeline, assembles the raw media into WAV/MP4 files, and hands them off to the slow brain worker for analysis.

## Features

- Real-time YAMNet audio classification with configurable label thresholds and cooldown logic.
- 20-second rolling pre-roll buffer (audio + video) so incidents capture context before the trigger.
- ZMQ-based transport (REQ/REP control, PUSH/PULL data) for decoupled edge-to-server communication.
- Whisper-based speech transcription with automatic backend selection (faster-whisper preferred).
- X-CLIP / CLIP zero-shot video scoring across five behavioral categories.
- Dynamic modality weighting when video evidence is weak or missing.
- Risk scoring (0--100) with compounding multi-category effects.
- Natural-language narrative generation for pgvector semantic search indexing.
- Cross-platform webcam and microphone auto-detection (Linux, macOS, Windows).

## Prerequisites

- **Python 3.12** (required by the TensorFlow 2.20 dependency).
- **uv** -- fast Python package manager. Install it from [docs.astral.sh/uv](https://docs.astral.sh/uv/).
- A working webcam and microphone (for live mode), or use `--demo-mode` for testing.
- Linux is the primary development platform. macOS and Windows are supported but less tested.

## Installation

```bash
git clone https://github.com/DaftOdyssey/bodycam-ai-capstone.git
cd bodycam-ai-capstone
uv sync
```

This installs all dependencies listed in `pyproject.toml` into a local `.venv`.

## How to Run

You need **two terminals**. Both commands must be run from the project root.

### Terminal 1 -- Start the test harness (server)

```bash
uv run python minimal_test_harness.py
```

This starts the ZMQ control and data listeners on `tcp://127.0.0.1:5555` and `tcp://127.0.0.1:5556`. It will wait for the edge pipeline to connect and send incident data.

### Terminal 2 -- Start the edge pipeline (bodycam)

```bash
uv run python edge_pipeline.py --demo-mode
```

The edge pipeline will:
1. Load the YAMNet model from TensorFlow Hub (first run downloads ~20 MB).
2. Auto-detect and open your webcam and microphone.
3. Begin buffering audio/video into a 20-second rolling window.
4. Listen for monitored audio events. When a trigger fires, it streams pre-roll and live media to the harness.

When an incident ends, the harness assembles the media and automatically runs the slow brain worker. Results are written to `events/<event_id>/slow_brain_output.json`.

### Useful CLI flags

| Flag | Description |
|---|---|
| `--demo-mode` | Print large trigger banners to the console. |
| `--video-device N` | Pin a specific webcam index instead of auto-detecting. |
| `--resolution 720p` | Use 1280x720 instead of the default 1920x1080. |
| `--cooldown N` | Seconds between triggers (default: 10). |
| `--rms-gate N` | Minimum audio RMS to run inference (default: 0.01). |

### Manual incident clear

While the harness is running, type `clear` (or `c`) and press Enter in Terminal 1 to manually signal the edge pipeline to end the active incident.

## Project Structure

```
bodycam-ai-capstone/
  pyproject.toml              # Project metadata and dependencies
  edge_pipeline.py            # Real-time edge audio/video pipeline
  slow_brain_worker.py        # Offline multimodal incident analysis
  minimal_test_harness.py     # ZMQ server bridging edge to slow brain
  README.md
  .gitignore
```

## License

This project is part of a university capstone and is provided as-is for educational and demonstration purposes.
