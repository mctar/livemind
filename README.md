# Live Mind Map

Real-time conversation visualization. Microphone audio is transcribed locally on-device, then AI extracts concepts and relationships into an animated, interactive mind map — all as people speak.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

As people talk in a meeting, workshop, or lecture, the system:

1. **Captures audio** from any microphone
2. **Transcribes locally** using Kyutai STT on Apple Silicon — nothing leaves your machine
3. **Extracts concepts** every 20 seconds via Claude, identifying key ideas and their relationships
4. **Renders a live map** — an animated force-directed graph that grows, evolves, and self-manages as the conversation unfolds

Concepts that stop being discussed fade out. Important ideas stick around. The map stays clean even in a 2-hour session.

## Quick start

```bash
# Clone
git clone https://github.com/mctar/livemind.git
cd livemind

# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Run
python app.py
```

Pick your microphone from the list. Open [localhost:8765](http://localhost:8765) in your browser.

## Pages

| URL | What |
|-----|------|
| [`/`](http://localhost:8765/) | Main mind map UI |
| [`/admin`](http://localhost:8765/admin) | Real-time monitoring dashboard |
| [`/doc`](http://localhost:8765/doc) | User documentation |

## Requirements

- **Mac with Apple Silicon** (M1/M2/M3/M4) — local STT runs on the Neural Engine
- **Python 3.11+**
- **Anthropic API key** — for concept extraction via Claude
- A microphone

## Architecture

```
app.py            FastAPI server — WebSocket + REST, Claude proxy, session persistence
stt_worker.py     Audio capture + VAD + Kyutai STT inference (background threads)
db.py             SQLite persistence (sessions, transcripts, snapshots, actions)
reconciler.py     Graph lifecycle — scoring, decay, budget enforcement
live-mindmap.html Frontend — D3.js force-directed graph + transcript sidebar
admin.html        Monitoring dashboard — STT, Claude, graph churn metrics
doc.html          User documentation
```

No frameworks, no Docker, no build step. Vanilla HTML/CSS/JS frontend served by FastAPI.

## Features

- **Local speech-to-text** — Kyutai STT on MLX, 24kHz mono, partial transcripts every ~320ms
- **Voice activity detection** — energy-based VAD with hysteresis skips silence automatically
- **Smart graph management** — max 24 active nodes, importance scoring (recency + frequency + centrality), automatic decay after 12 min
- **Node interactions** — right-click to pin, hide, rename, merge, or promote concepts
- **Session persistence** — all data stored in SQLite; reconnect and restore from where you left off
- **Circuit breaker** — exponential backoff on API errors, graceful degradation (transcript keeps flowing)
- **Admin dashboard** — real-time charts for STT performance, audio buffer, Claude latency, graph churn, frontend FPS

## Session management

Sessions store everything: transcripts, graph snapshots, user actions. Use **New Session** to archive the current session and start fresh. All past session data remains in `livemind.db`.

To wipe all history: delete `livemind.db` and restart.

## Configuration

Key settings in `live-mindmap.html` (the `C` object):

| Setting | Default | Description |
|---------|---------|-------------|
| `interval` | 20000 | Claude analysis interval (ms) |
| `minLen` | 50 | Min new chars before triggering analysis |
| `maxN` | 30 | Max nodes in Claude prompt |
| `model` | `claude-sonnet-4-20250514` | Claude model |

Server flags: `python app.py -d <device> --host 0.0.0.0 --port 8765`

## License

MIT
