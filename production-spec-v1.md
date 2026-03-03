# Live Mind Map Production Spec v1

Date: March 3, 2026
Owner: Live Mind Map team
Status: Draft for implementation

## 1. Objective

Make the current prototype production-grade for live 2-3 hour meetings:
- Stable operation for full meeting duration.
- Readable and actionable mind map throughout.
- Recoverable sessions after browser refresh or brief network loss.
- Secure key handling (no provider keys in browser).

## 2. Success Criteria and SLOs

- Session uptime: >= 99.5% over a continuous 3-hour run.
- Transcript-to-graph update latency: p95 <= 5s.
- Restore latency after reconnect/refresh: <= 10s.
- Readability budget: <= 24 active nodes, <= 30 active edges.
- Data durability: no transcript loss on client refresh/crash.
- Security: LLM provider keys never exposed in browser traffic.

## 3. Current Gaps

- LLM is called directly from browser using API key.
- Full transcript is sent repeatedly to LLM, causing latency/cost/context drift over long meetings.
- Graph/transcript state is browser-memory only.
- No job queue/backpressure for analysis.
- STT chunking is optimized for throughput, not low latency.

## 4. Target Architecture

Components:
- `stt-worker` (Python): audio capture + STT + segment emit.
- `api-server` (FastAPI): sessions, auth, websocket fanout, persistence, LLM proxy.
- `analysis-worker`: queued LLM analysis + deterministic graph reconciler.
- `db` (SQLite first, Postgres-ready): transcript, graph state, snapshots, user actions.
- `frontend` (existing HTML/JS): visualization + controls; consumes server stream.

Data flow:
1. STT emits transcript segments with sequence numbers.
2. API stores segments and pushes them over websocket.
3. Analysis worker consumes latest unsent range and updates graph deltas.
4. API broadcasts graph deltas/snapshots.
5. Frontend applies deltas with stable layout constraints.

## 5. Session and Persistence Model

- Session ID created at meeting start.
- Monotonic `seq` per transcript segment.
- Graph snapshots stored every 60s and on major user actions.
- Reconnect protocol restores latest snapshot plus missed events from `seq`.
- Session close generates final summary and export bundle.

## 6. Readability and Usability Requirements

Hard budgets:
- Active map: max 24 nodes, max 30 edges.
- Label format: 2-4 words, title case.
- Edge label: 1-2 words.

Aging and lifecycle:
- Node state: `active`, `parked`, `archived`, `hidden`.
- Decay to `parked` after 12 minutes without supporting mentions.
- Reactivate if mentioned >= 2 times within 3 minutes.

Scoring:
- Importance score:
  - `0.45 * recency`
  - `0.35 * mention_frequency`
  - `0.20 * centrality`
  - `+ pin_bonus`
- Edge weight score uses co-mention recency and relation confidence.

UX controls:
- `Pin`, `Merge`, `Hide`, `Rename`, `Promote`.
- Dedicated lanes for `Decision`, `Action`, `Open Question`.
- Filter by time window/category/speaker and search-by-concept.

Layout stability:
- Category anchors fixed in soft zones.
- Per-update position clamp (for example 30px max movement).
- Avoid global re-layout on small deltas to reduce visual thrash.

## 7. Analysis Pipeline (Incremental, not full replay)

Use three context layers per analysis job:
1. Fresh transcript window (for example last 2-4 minutes).
2. Rolling micro-summary blocks.
3. Current canonical graph state (IDs, scores, states).

Rules:
- LLM proposes graph changes only.
- Deterministic reconciler enforces limits, ID stability, and decay policy.
- If queue is backed up, coalesce stale jobs and process only newest range.

## 8. Reliability and Failure Modes

- Websocket heartbeat every 5s.
- Cursor-based resume from last acknowledged `seq`.
- Circuit breaker for LLM 429/5xx with exponential backoff.
- Degraded mode:
  - Continue transcript streaming.
  - Hold last stable graph.
  - Show status banner to user.
- Bounded queues and memory caps to prevent runaway growth.

## 9. Security and Compliance

- Move all LLM API calls server-side.
- Store provider keys in server env/secrets manager only.
- Browser receives short-lived session token; no direct provider auth.
- Audit log for user actions (`pin`, `merge`, `hide`, etc.).
- Optional PII redaction pass before persistence/export.

## 10. Observability

Core metrics:
- `stt_latency_ms` (first partial, final segment)
- `analysis_latency_ms`
- `analysis_queue_depth`
- `graph_node_churn_per_min`
- `graph_edge_churn_per_min`
- `ws_reconnect_count`
- `restore_duration_ms`
- `frontend_fps_avg`

Alert examples:
- p95 analysis latency > 8s for 5 min.
- queue depth > 3 for 2 min.
- reconnect count > 5 per 10 min.
- node churn > 12/min for 10 min.

## 11. STT Low-Latency Optimization Plan

### 11.1 Baseline issue in current implementation

Current STT settings use:
- `CHUNK_SECONDS = 8`
- `OVERLAP_SECONDS = 1`
- 1s polling cadence

This naturally produces multi-second latency bursts and can approach ~8-9s before text appears.

### 11.2 Target latency

- First partial transcript: <= 1.0s p95.
- Stable finalized segment: <= 2.5s p95.

### 11.3 Quick wins (high impact, low risk)

1. Reduce chunk size and stride:
- Chunk: 2.0s (from 8.0s)
- Stride: 0.5-1.0s
- Keep overlap/dedup logic.

2. Emit partial hypotheses:
- Stream decoder output every ~200-300ms.
- Mark messages as `partial` then `final`.

3. Replace polling loop with event-triggered processing:
- Trigger decode when stride-sized audio arrives.
- Remove fixed 1s wait loop where possible.

4. Lower capture blocksize:
- Use ~20-40ms audio callbacks instead of ~100ms.

### 11.4 Medium-term optimizations

1. Voice activity detection (VAD):
- Skip silence windows.
- Flush decode immediately at speech end boundaries.

2. Faster resampling path:
- Avoid heavy per-callback interpolation overhead.
- Use a streaming resampler designed for real-time audio.

3. Model/runtime tuning:
- Keep model warm and pinned to dedicated worker thread/process.
- Evaluate quantized checkpoints (`q8`, then `q4` if quality acceptable).

4. Double-buffer pipeline:
- Capture, decode, and websocket send in separate bounded queues.
- Prevent decode stalls from blocking capture.

### 11.5 Guardrails for quality

- Keep overlap dedup logic to avoid duplicate phrase emissions.
- Add final-pass correction on `final` segments to reduce partial noise.
- Track WER proxy via replay tests against reference transcripts.

## 12. API Contract Summary

REST:
- `POST /v1/sessions`
- `POST /v1/sessions/{id}/segments`
- `POST /v1/sessions/{id}/actions`
- `GET /v1/sessions/{id}/restore?from_seq=...`
- `POST /v1/sessions/{id}/end`

Websocket events:
- `transcript.segment` (`partial`/`final`)
- `graph.delta`
- `graph.snapshot`
- `status.health`
- `status.error`

## 13. Implementation Plan (10 days)

Day 1:
- FastAPI scaffold + session lifecycle + secure key config.

Day 2:
- DB schema + migrations + repositories.

Day 3:
- Transcript ingest endpoint + websocket stream + heartbeat.
- STT quick-win pass:
  - reduce chunking to low-latency defaults (target: 2s chunk, 0.5-1.0s stride)
  - reduce capture blocksize to ~20-40ms
  - switch decode trigger from fixed polling to event-driven buffering

Day 4:
- Server-side LLM proxy + queue + retries/backoff.

Day 5:
- Incremental context builder + periodic snapshots.
- STT streaming improvements:
  - emit `partial` transcript events every ~200-300ms
  - emit `final` events with boundary dedup/cleanup

Day 6:
- Deterministic reconciler + readability budgets + decay lifecycle.
- STT quality/speed tuning:
  - add VAD-based silence skipping and end-of-speech flush
  - evaluate quantized runtime options (`q8`, optional `q4`) against quality guardrails

Day 7:
- Frontend protocol migration + restore on reload.

Day 8:
- User controls (`pin`, `merge`, `hide`, `rename`, `promote`) + parked panel.

Day 9:
- 3-hour transcript replay soak tests + metrics dashboards.
- Validate STT latency targets (partial and final p95) using replay and live dry run.

Day 10:
- Hardening, runbook, release checklist, go/no-go review.

## 14. Definition of Done

- 3-hour soak test passes without crash/restart.
- p95 end-to-end latency meets target.
- Map remains within active budgets and is usable by operator.
- Refresh/reconnect fully restores session in <= 10s.
- No provider key exposure in browser.
