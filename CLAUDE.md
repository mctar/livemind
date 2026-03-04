# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Live Mind Map: a real-time meeting/conversation visualizer. Microphone audio → local STT (Kyutai/Moshi on MLX) → WebSocket → browser → Claude API → D3.js force-directed graph of concepts and relationships.

## Running the Project

```bash
source .venv/bin/activate
python app.py -d <device_index>

# Or interactive device picker:
python app.py
```

Server starts at `http://localhost:8765`. Open `/` for the main UI, `/admin` for the monitoring dashboard. No build step needed — the frontend is served by FastAPI.

## Architecture

```
app.py              — FastAPI server: WS + REST routes, Claude proxy, broadcast/snapshot loops
stt_worker.py       — Audio capture + VAD + Kyutai STT inference (background threads)
db.py               — SQLite persistence (sessions, segments, snapshots, actions)
reconciler.py       — Deterministic graph reconciler (scoring, decay, budget enforcement)
live-mindmap.html   — Frontend: D3.js mind map + transcript sidebar + context menu
admin.html          — Monitoring dashboard: STT, Claude, graph churn metrics
requirements.txt    — Python dependencies
```

**`app.py`** — FastAPI server (`http://localhost:8765`)
- Lifespan: initializes DB, starts STT pipeline, spawns broadcast + snapshot loops
- `GET /` serves `live-mindmap.html`, `GET /admin` serves `admin.html`
- `WS /ws`: transcript streaming, Claude proxy, session reconnect, metrics
- `POST /v1/sessions`: create session in SQLite
- `GET /v1/sessions/{id}/restore?from_seq=N`: snapshot + segments since N
- `POST /v1/sessions/{id}/actions`: pin/hide/rename/merge/promote → reconciler
- `GET /v1/metrics`: REST polling for admin metrics
- Claude proxy runs reconciler on responses; stores snapshots in DB

**`stt_worker.py`** — Audio capture + STT
- `VAD` class: energy-based voice activity detection with hysteresis
- `select_input_device()`: interactive mic picker
- `start_stt_pipeline()`: spawns capture + STT threads, returns shutdown event
- Captures at native rate, resamples to 24kHz, emits partials every ~320ms

**`db.py`** — SQLite with WAL mode
- Tables: `sessions`, `segments`, `snapshots`, `actions`
- All async via `aiosqlite`
- DB file: `livemind.db` (auto-created on first run)

**`reconciler.py`** — Graph lifecycle management
- Node states: active → parked (12min decay) → archived/hidden
- Scoring: `0.45*recency + 0.35*frequency + 0.20*centrality + pin_bonus`
- Budget: max 24 active nodes, parks lowest-scoring non-pinned
- User actions: pin, hide, rename, merge, promote

## Key Configuration (in `live-mindmap.html`)

Located in the `C` object at the top of the `<script>` block:
- `C.interval`: how often Claude is called in ms (default 20000)
- `C.minLen`: minimum new chars before triggering analysis (default 50)
- `C.maxN`: max node cap enforced in Claude prompt (default 30)
- `C.model`: Claude model string
- `C.colors`: 10-element palette array; index maps to `G.cmap` by group name

## Frontend Internals

The global `G` object holds all mutable state. Key fields:
- `G.txt`: full accumulated transcript string
- `G.sent`: character offset — only `G.txt.slice(G.sent)` is sent as "new segment" each Claude call
- `G.nodes` / `G.edges`: current graph state fed into D3 simulation
- `G.cmap` / `G.ci`: group→color assignment, persists across graph updates
- `G.sessionId` / `G.lastSeq`: server session tracking for reconnect
- `G.mergeSource`: merge mode state for node merging
- `G._ctxNode`: context menu target node

Context menu: right-click a node for Pin/Unpin, Promote, Rename, Merge, Hide.

STT fallback: if the STT server URL is left blank (or the WebSocket fails), `fallbackMic()` activates the browser's Web Speech API.

FPS tracking: reports `frontend_metrics` over WS every 5s.

## WebSocket Message Protocol

Server → browser:
- `{"type":"transcript","text":"...","seq":N,"timestamp":T}` — final transcript
- `{"type":"partial_transcript","text":"...","seq":N,"timestamp":T}` — partial
- `{"type":"claude_response","status":200,"data":{...},"req_id":"..."}` — Claude result (reconciled)
- `{"type":"graph_update","graph":{...}}` — graph update from user action
- `{"type":"restore","snapshot":{...},"segments":[...],"restore_ms":N}` — session restore
- `{"type":"status","status":"connected","message":"..."}` — connection status
- `{"type":"metrics",...}` — metrics response
- `{"type":"pong"}` — keepalive

Browser → server:
- `{"type":"ping"}` — keepalive
- `{"type":"get_metrics"}` — request metrics
- `{"type":"claude_request","req_id":"...","body":{...}}` — Claude API proxy
- `{"type":"connect_session","session_id":"...","last_seq":N}` — reconnect with cursor
- `{"type":"frontend_metrics","fps":N}` — FPS report

## Troubleshooting Audio

- **macOS mic permission**: System Settings > Privacy & Security > Microphone > enable for Terminal
- **Mic input level**: System Settings > Sound > Input — turn up input volume if RMS stays below 0.01 when speaking
- The server prints RMS every 5s; speech should read ~0.05–0.2.
- Audio is captured at the device's native sample rate and resampled to 24kHz internally.

## Python Dependencies

Defined in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

Key packages: `fastapi`, `uvicorn`, `aiosqlite`, `aiohttp`, `moshi_mlx`, `sounddevice`, `numpy`, `huggingface-hub`, `sentencepiece`
