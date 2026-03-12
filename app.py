#!/usr/bin/env python3
"""
Live Mind Map — FastAPI Server
Replaces stt_server.py. Handles WebSocket streaming, REST API,
Claude proxy, session persistence, and graph reconciliation.
"""

import asyncio, json, time, threading, queue, sys, argparse, os, uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import aiohttp

import db
from stt_worker import select_input_device, start_stt_pipeline
from reconciler import GraphReconciler

# ─── Load .env ───
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─── Global state ───
transcript_queue: queue.Queue = queue.Queue()
connected_clients: set[WebSocket] = set()
client_sessions: dict[WebSocket, str] = {}  # ws → session_id

metrics = {
    "started_at": time.time(),
    "chunks_processed": 0,
    "chunks_skipped_silent": 0,
    "chunks_skipped_catchup": 0,
    "stt_last_duration": 0.0,
    "stt_avg_duration": 0.0,
    "stt_total_time": 0.0,
    "stt_last_text": "",
    "stt_empty_results": 0,
    "stt_partials_emitted": 0,
    "audio_buffer_seconds": 0.0,
    "audio_rms": 0.0,
    "tokenizer_recreations": 0,
    "tokenizer_last_ms": 0.0,
    "stt_e2e_last": 0.0,
    "stt_e2e_avg": 0.0,
    "stt_e2e_total": 0.0,
    "claude_calls": 0,
    "claude_errors": 0,
    "claude_last_duration": 0.0,
    "claude_avg_duration": 0.0,
    "claude_total_time": 0.0,
    "ws_clients": 0,
    "transcript_queue_size": 0,
    "chunk_seconds": 2,
    "cb_state": "closed",
    "cb_failures": 0,
    "vad_state": "silent",
    "ws_reconnects": 0,
    "last_restore_ms": 0.0,
    "frontend_fps": 0.0,
    "nodes_added_per_min": 0,
    "nodes_removed_per_min": 0,
    "edge_churn_per_min": 0,
    "analysis_queue_depth": 0,
    "claude_last_error": "",
}
metrics_lock = threading.Lock()

# Circuit breaker
cb_state = "closed"
cb_failures = 0
cb_backoff_until = 0.0
cb_backoff_secs = 5.0
CB_MAX_BACKOFF = 60.0
CB_FAILURE_THRESHOLD = 3
cb_lock = threading.Lock()

# Monotonic sequence counter for transcript messages
_seq_counter = 0
_seq_lock = threading.Lock()

# Graph reconciler
reconciler = GraphReconciler()
_current_session_id: str | None = None
_summary: str = ""

# Store device_idx from CLI
_device_idx: int | None = None


def _next_seq() -> int:
    global _seq_counter
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


# ─── Lifespan ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    start_stt_pipeline(_device_idx, transcript_queue, metrics, metrics_lock)
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(snapshot_loop())
    print(f"Server ready: http://localhost:{WS_PORT}")
    print(f"  Main UI:  http://localhost:{WS_PORT}/")
    print(f"  Admin:    http://localhost:{WS_PORT}/admin\n")
    yield
    await db.close_db()


app = FastAPI(lifespan=lifespan)
WS_PORT = 8765


# ─── Static serving ───
@app.get("/", response_class=HTMLResponse)
async def serve_main():
    return FileResponse(os.path.join(os.path.dirname(__file__), "live-mindmap.html"))


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    return FileResponse(os.path.join(os.path.dirname(__file__), "admin.html"))


@app.get("/doc", response_class=HTMLResponse)
async def serve_doc():
    return FileResponse(os.path.join(os.path.dirname(__file__), "doc.html"))


# ─── REST endpoints ───
@app.post("/v1/sessions")
async def create_session(request: Request):
    body = await request.json()
    session_id = str(uuid.uuid4())[:8]
    topic = body.get("topic", "")
    session = await db.create_session(session_id, topic)
    global _current_session_id
    _current_session_id = session_id
    return JSONResponse(session)


@app.post("/v1/sessions/{session_id}/end")
async def end_session(session_id: str, request: Request):
    """End a session: flush final snapshot, store summary, reset reconciler."""
    global _current_session_id, _summary, _seq_counter
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    summary = body.get("summary", _summary)
    # Final snapshot
    if reconciler.nodes:
        await db.store_snapshot(session_id, _seq_counter, reconciler.get_full_state(), "end")
    await db.end_session(session_id, summary)
    # Reset server state
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()
    _summary = ""
    with _seq_lock:
        _seq_counter = 0
    while not transcript_queue.empty():
        try:
            transcript_queue.get_nowait()
        except Exception:
            break
    if _current_session_id == session_id:
        _current_session_id = None
    return JSONResponse({"ok": True, "session_id": session_id})


@app.get("/v1/sessions/{session_id}/restore")
async def restore_session(session_id: str, from_seq: int = 0):
    t0 = time.time()
    snapshot = await db.get_latest_snapshot(session_id)
    segments = await db.get_segments_since(session_id, from_seq)
    restore_ms = (time.time() - t0) * 1000
    with metrics_lock:
        metrics["last_restore_ms"] = restore_ms
    return JSONResponse({
        "snapshot": snapshot,
        "segments": segments,
        "restore_ms": round(restore_ms, 1),
    })


@app.post("/v1/sessions/{session_id}/actions")
async def session_action(session_id: str, request: Request):
    body = await request.json()
    action_type = body.get("action")
    payload = body.get("payload", {})
    await db.store_action(session_id, action_type, payload)
    graph = reconciler.apply_action(action_type, payload)
    # Broadcast updated graph to all connected clients
    msg = json.dumps({"type": "graph_update", "graph": graph})
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse({"ok": True, "graph": graph})


@app.post("/v1/sessions/new")
async def new_session(request: Request):
    """End current session (if any) and start a fresh one. Returns the new session."""
    global _current_session_id, _summary, _seq_counter
    body = await request.json() if request.headers.get("content-length", "0") != "0" else {}
    topic = body.get("topic", "")
    # End current session
    if _current_session_id:
        if reconciler.nodes:
            await db.store_snapshot(_current_session_id, _seq_counter, reconciler.get_full_state(), "end")
        await db.end_session(_current_session_id, _summary)
    # Reset all state
    reconciler.nodes.clear()
    reconciler.edges.clear()
    reconciler._mention_log.clear()
    reconciler._churn_log.clear()
    _summary = ""
    with _seq_lock:
        _seq_counter = 0
    # Drain any leftover transcript messages from STT queue
    while not transcript_queue.empty():
        try:
            transcript_queue.get_nowait()
        except Exception:
            break
    # Create new
    session_id = str(uuid.uuid4())[:8]
    session = await db.create_session(session_id, topic)
    _current_session_id = session_id
    # Notify all connected frontends
    msg = json.dumps({"type": "session_reset", "session_id": session_id})
    for ws in list(connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            pass
    return JSONResponse(session)


@app.get("/v1/metrics")
async def get_metrics_rest():
    with metrics_lock:
        m = {**metrics, "uptime": time.time() - metrics["started_at"]}
    churn = reconciler.get_churn_metrics()
    m.update(churn)
    m["current_session_id"] = _current_session_id
    m["active_nodes"] = len([ns for ns in reconciler.nodes.values() if ns.state == "active"])
    return JSONResponse(m)


# ─── Claude proxy ───
async def _proxy_claude(websocket: WebSocket, req: dict):
    """Proxy a Claude API request server-side with circuit breaker."""
    global cb_state, cb_failures, cb_backoff_until, cb_backoff_secs, _summary

    with cb_lock:
        now = time.time()
        if cb_state == "open":
            if now < cb_backoff_until:
                wait = cb_backoff_until - now
                print(f"  Claude: circuit breaker OPEN, retry in {wait:.0f}s", file=sys.stderr)
                with metrics_lock:
                    metrics["cb_state"] = "open"
                await websocket.send_json({
                    "type": "claude_response",
                    "status": 503,
                    "data": {"error": "Circuit breaker open", "retry_after": wait},
                    "req_id": req.get("req_id"),
                })
                return
            else:
                cb_state = "half_open"
                print("  Claude: circuit breaker half_open, testing...")
                with metrics_lock:
                    metrics["cb_state"] = "half_open"

    t0 = time.time()
    try:
        body = req.get("body", {})
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json=body,
            ) as resp:
                data = await resp.json()
                dt = time.time() - t0
                with metrics_lock:
                    metrics["claude_calls"] += 1
                    metrics["claude_last_duration"] = dt
                    metrics["claude_total_time"] += dt
                    metrics["claude_avg_duration"] = metrics["claude_total_time"] / metrics["claude_calls"]

                if resp.status == 200:
                    print(f"  Claude: 200 OK ({dt:.1f}s)")
                    with cb_lock:
                        cb_state = "closed"
                        cb_failures = 0
                        cb_backoff_secs = 5.0
                    with metrics_lock:
                        metrics["cb_state"] = "closed"
                        metrics["cb_failures"] = 0
                        metrics["claude_last_error"] = ""

                    # Run reconciler on Claude's response
                    try:
                        raw_text = "".join(c.get("text", "") for c in data.get("content", []))
                        parsed = json.loads(raw_text.replace("```json", "").replace("```", "").strip())
                        if parsed.get("nodes") and parsed.get("edges") is not None:
                            graph = reconciler.reconcile(parsed)
                            if parsed.get("summary"):
                                _summary = parsed["summary"]
                            if _current_session_id:
                                await db.store_snapshot(
                                    _current_session_id, _seq_counter,
                                    reconciler.get_full_state(), "analysis"
                                )
                            data = {
                                "content": [{"type": "text", "text": json.dumps({
                                    **graph, "summary": _summary,
                                })}],
                            }
                            churn = reconciler.get_churn_metrics()
                            with metrics_lock:
                                metrics["nodes_added_per_min"] = churn["nodes_added_per_min"]
                                metrics["nodes_removed_per_min"] = churn["nodes_removed_per_min"]
                                metrics["edge_churn_per_min"] = churn["edge_churn_per_min"]
                    except (json.JSONDecodeError, KeyError) as parse_err:
                        print(f"  Claude: response parse error: {parse_err}", file=sys.stderr)
                        print(f"  Claude: raw text: {raw_text[:200]}", file=sys.stderr)

                elif resp.status == 429:
                    err_msg = data.get("error", {}).get("message", "Rate limited")
                    print(f"  Claude: 429 RATE LIMITED ({dt:.1f}s) — {err_msg}", file=sys.stderr)
                    with cb_lock:
                        cb_state = "open"
                        cb_backoff_secs = min(cb_backoff_secs * 2, CB_MAX_BACKOFF)
                        cb_backoff_until = time.time() + cb_backoff_secs
                        cb_failures += 1
                    with metrics_lock:
                        metrics["claude_errors"] += 1
                        metrics["cb_state"] = "open"
                        metrics["cb_failures"] = cb_failures
                        metrics["claude_last_error"] = f"429: {err_msg}"

                elif resp.status >= 500:
                    err_msg = data.get("error", {}).get("message", f"Server error {resp.status}")
                    print(f"  Claude: {resp.status} SERVER ERROR ({dt:.1f}s) — {err_msg}", file=sys.stderr)
                    with cb_lock:
                        cb_failures += 1
                        if cb_failures >= CB_FAILURE_THRESHOLD:
                            cb_state = "open"
                            cb_backoff_secs = min(cb_backoff_secs * 2, CB_MAX_BACKOFF)
                            cb_backoff_until = time.time() + cb_backoff_secs
                    with metrics_lock:
                        metrics["claude_errors"] += 1
                        metrics["cb_state"] = cb_state
                        metrics["cb_failures"] = cb_failures
                        metrics["claude_last_error"] = f"{resp.status}: {err_msg}"

                else:
                    err_msg = data.get("error", {}).get("message", f"HTTP {resp.status}")
                    print(f"  Claude: {resp.status} ERROR ({dt:.1f}s) — {err_msg}", file=sys.stderr)
                    with metrics_lock:
                        metrics["claude_errors"] += 1
                        metrics["claude_last_error"] = f"{resp.status}: {err_msg}"

                await websocket.send_json({
                    "type": "claude_response",
                    "status": resp.status,
                    "data": data,
                    "req_id": req.get("req_id"),
                })
    except Exception as e:
        dt = time.time() - t0
        print(f"  Claude: EXCEPTION ({dt:.1f}s) — {type(e).__name__}: {e}", file=sys.stderr)
        with cb_lock:
            cb_failures += 1
            if cb_failures >= CB_FAILURE_THRESHOLD:
                cb_state = "open"
                cb_backoff_secs = min(cb_backoff_secs * 2, CB_MAX_BACKOFF)
                cb_backoff_until = time.time() + cb_backoff_secs
        with metrics_lock:
            metrics["claude_calls"] += 1
            metrics["claude_errors"] += 1
            metrics["cb_state"] = cb_state
            metrics["cb_failures"] = cb_failures
            metrics["claude_last_error"] = f"{type(e).__name__}: {e}"
        await websocket.send_json({
            "type": "claude_response",
            "status": 500,
            "data": {"error": str(e)},
            "req_id": req.get("req_id"),
        })


# ─── WebSocket ───
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    with metrics_lock:
        metrics["ws_clients"] = len(connected_clients)
    print(f"Browser connected ({len(connected_clients)} clients)")
    await websocket.send_json({
        "type": "status", "status": "connected",
        "message": "STT server ready",
    })
    try:
        while True:
            raw = await websocket.receive_text()
            d = json.loads(raw)
            msg_type = d.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "get_metrics":
                with metrics_lock:
                    m = {**metrics, "uptime": time.time() - metrics["started_at"]}
                churn = reconciler.get_churn_metrics()
                m.update(churn)
                m["current_session_id"] = _current_session_id
                m["active_nodes"] = len([ns for ns in reconciler.nodes.values() if ns.state == "active"])
                await websocket.send_json({"type": "metrics", **m})

            elif msg_type == "claude_request":
                asyncio.create_task(_proxy_claude(websocket, d))

            elif msg_type == "connect_session":
                session_id = d.get("session_id")
                last_seq = d.get("last_seq", 0)
                if session_id:
                    client_sessions[websocket] = session_id
                    with metrics_lock:
                        metrics["ws_reconnects"] += 1
                    # Send restore data
                    t0 = time.time()
                    snapshot = await db.get_latest_snapshot(session_id)
                    segments = await db.get_segments_since(session_id, last_seq)
                    restore_ms = (time.time() - t0) * 1000
                    with metrics_lock:
                        metrics["last_restore_ms"] = restore_ms
                    await websocket.send_json({
                        "type": "restore",
                        "snapshot": snapshot,
                        "segments": segments,
                        "restore_ms": round(restore_ms, 1),
                    })

            elif msg_type == "frontend_metrics":
                fps = d.get("fps", 0)
                with metrics_lock:
                    metrics["frontend_fps"] = fps

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        connected_clients.discard(websocket)
        client_sessions.pop(websocket, None)
        with metrics_lock:
            metrics["ws_clients"] = len(connected_clients)
        print(f"Browser disconnected ({len(connected_clients)} clients)")


# ─── Background loops ───
async def broadcast_loop():
    """Poll transcript_queue and broadcast to all WS clients."""
    while True:
        try:
            msg = transcript_queue.get_nowait()
            seq = _next_seq()
            msg["seq"] = seq

            # Persist segment
            if _current_session_id and msg.get("type") in ("transcript", "partial_transcript"):
                await db.store_segment(
                    _current_session_id, seq, msg["text"],
                    is_partial=(msg["type"] == "partial_transcript"),
                    timestamp=msg.get("timestamp", time.time()),
                )

            if connected_clients:
                p = json.dumps(msg)
                for ws in list(connected_clients):
                    try:
                        await ws.send_text(p)
                    except Exception:
                        pass
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


async def snapshot_loop():
    """Periodic graph snapshot every 60s."""
    while True:
        await asyncio.sleep(60)
        if _current_session_id and reconciler.nodes:
            try:
                await db.store_snapshot(
                    _current_session_id, _seq_counter,
                    reconciler.get_full_state(), "periodic"
                )
            except Exception as e:
                print(f"  Snapshot error: {e}", file=sys.stderr)


# ─── Entry point ───
if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(description="Live Mind Map Server")
    p.add_argument("-d", "--device", type=int, default=None,
                   help="Input device index (skips interactive picker)")
    p.add_argument("--host", default="localhost", help="Bind host")
    p.add_argument("--port", type=int, default=8765, help="Bind port")
    args = p.parse_args()

    print("=" * 50)
    print("  Live Mind Map : Server")
    print("=" * 50 + "\n")

    _device_idx = select_input_device(forced=args.device)
    WS_PORT = args.port

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
