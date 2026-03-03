# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Live Mind Map: a real-time meeting/conversation visualizer. Microphone audio → local STT (Kyutai/Moshi on MLX) → WebSocket → browser → Claude API → D3.js force-directed graph of concepts and relationships.

## Running the Project

```bash
# Start the STT backend (must be running before browser connects)
source .venv/bin/activate
python3 stt_server.py

# Frontend: open live-mindmap.html directly in a browser (no build step)
```

There is no build system. The frontend is a single self-contained HTML file using CDN-hosted D3.js. The Python backend uses the `.venv` virtual environment.

## Architecture

Two loosely coupled components:

**`stt_server.py`** — Python WebSocket server (`ws://localhost:8765`)
- Captures microphone audio at 24kHz mono via `sounddevice`
- Runs Kyutai STT inference (`moshi_mlx`) on 8-second chunks with 1-second overlap
- `_overlap()` deduplicates transcript segments at chunk boundaries
- Broadcasts transcribed text to all connected browser clients

**`live-mindmap.html`** — Single-page frontend
- Setup panel collects Claude API key, meeting topic, and STT server URL
- Connects to STT server via WebSocket; streams transcript text into sidebar
- Every 20 seconds, sends accumulated transcript to Claude (`claude-sonnet-4-20250514`) via direct REST call from the browser
- Claude returns a JSON graph structure (nodes + edges, max 30 nodes)
- D3.js force-directed graph animates concept births, deaths, and repositioning
- Node colors encode concept category (10-color palette defined at top of script)

## Key Configuration (in `live-mindmap.html`)

Located at the top of the `<script>` block:
- `ANALYSIS_INTERVAL`: how often Claude is called (default 20s)
- `MIN_TRANSCRIPT_LENGTH`: minimum chars before triggering analysis (default 50)
- `MAX_NODES`: cap enforced in the Claude prompt (default 30)
- `MODEL`: Claude model string

## Troubleshooting Audio

- **macOS mic permission**: System Settings > Privacy & Security > Microphone > enable for Terminal
- **Mic input level**: System Settings > Sound > Input — turn up input volume if RMS stays below 0.01 when speaking
- `stt_server.py` prints RMS every 5s; speech should read ~0.05–0.2. Values near 0 mean audio isn't reaching the process.
- The server captures at the device's native sample rate (typically 44100 Hz) and resamples to 24000 Hz internally — do not change `SAMPLE_RATE` in the InputStream call.

## Python Dependencies

All installed in `.venv`:
- `moshi_mlx` — Kyutai STT model (MLX, Apple Silicon)
- `sounddevice` — audio capture
- `websockets` — WebSocket server
- `numpy` — audio buffer processing
- `huggingface_hub` — model download on first run
