#!/usr/bin/env python3
"""
Live Mind Map - STT Server
Captures mic, transcribes with Kyutai STT MLX, streams to browser via WebSocket.
"""

import asyncio, json, time, threading, queue, sys, argparse, os
import numpy as np

# Load .env file
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SECONDS = 3
OVERLAP_SECONDS = 0
WS_HOST = "localhost"
WS_PORT = 8765
MIN_RMS = 0.01

transcript_queue = queue.Queue()
audio_buffer = []
audio_lock = threading.Lock()
is_running = True
connected_clients = set()

# Metrics for admin panel
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
    "audio_buffer_seconds": 0.0,
    "audio_rms": 0.0,
    "tokenizer_recreations": 0,
    "ws_clients": 0,
    "transcript_queue_size": 0,
}
metrics_lock = threading.Lock()


def select_input_device(forced=None):
    import sounddevice as sd
    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        print("No input devices found.", file=sys.stderr)
        sys.exit(1)

    if forced is not None:
        dev = sd.query_devices(forced, "input")
        print(f"  Using device {forced}: {dev['name']}\n")
        return forced

    default_in = sd.default.device[0] if hasattr(sd.default.device, '__len__') else sd.default.device
    print("Available input devices:\n")
    for pos, (idx, d) in enumerate(inputs):
        marker = " *" if idx == default_in else "  "
        print(f"  [{pos}]{marker} {d['name']}  ({int(d['default_samplerate'])}Hz, {d['max_input_channels']}ch in)")
    default_pos = next((p for p, (i, _) in enumerate(inputs) if i == default_in), 0)

    if len(inputs) == 1:
        print(f"\n  Only one input — using [{0}] {inputs[0][1]['name']}\n")
        return inputs[0][0]

    try:
        raw = input(f"\nSelect device [{default_pos}]: ").strip()
        pos = int(raw) if raw else default_pos
        if 0 <= pos < len(inputs):
            chosen = inputs[pos][0]
        else:
            print(f"  Invalid, using default.")
            chosen = inputs[default_pos][0]
    except (ValueError, EOFError):
        chosen = inputs[default_pos][0]
    print()
    return chosen


def audio_capture_thread(device_idx):
    import sounddevice as sd
    try:
        dev = sd.query_devices(device_idx, "input")
        native_rate = int(dev["default_samplerate"])
        resample_ratio = SAMPLE_RATE / native_rate
        print(f"  Mic: {dev['name']} ({native_rate}Hz → {SAMPLE_RATE}Hz)")

        rms_report = {"last": 0.0}

        def cb(indata, frames, t, status):
            if status:
                print(f"  Audio status: {status}", file=sys.stderr)
            data = indata[:, 0]
            # Resample from native rate to target SAMPLE_RATE
            if native_rate != SAMPLE_RATE:
                n_out = max(1, int(len(data) * resample_ratio))
                data = np.interp(
                    np.linspace(0, len(data) - 1, n_out),
                    np.arange(len(data)),
                    data
                ).astype(np.float32)
            rms = float(np.sqrt(np.mean(data ** 2)))
            with metrics_lock:
                metrics["audio_rms"] = rms
            now = time.time()
            if now - rms_report["last"] > 5.0:
                print(f"  Audio RMS: {rms:.4f} ({'OK' if rms > MIN_RMS else 'silent/quiet'})")
                rms_report["last"] = now
            with audio_lock:
                audio_buffer.extend(data.tolist())

        with sd.InputStream(samplerate=native_rate, channels=CHANNELS,
                            dtype="float32", callback=cb, device=device_idx,
                            blocksize=int(native_rate * 0.1)):
            while is_running:
                time.sleep(0.1)
    except Exception as e:
        print(f"  AUDIO CAPTURE FAILED: {e}", file=sys.stderr)
        print("  Check: System Settings > Privacy & Security > Microphone > Terminal", file=sys.stderr)


def stt_thread():
    print("Loading Kyutai STT model...")
    try:
        import json
        import mlx.core as mx
        import mlx.nn as nn
        import rustymimi
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi_mlx import models, utils

        HF_REPO = "kyutai/stt-1b-en_fr-mlx"
        cfg_path = hf_hub_download(HF_REPO, "config.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)

        stt_cfg = cfg_dict.get("stt_config", None)
        mimi_path  = hf_hub_download(HF_REPO, cfg_dict["mimi_name"])
        model_path = hf_hub_download(HF_REPO, cfg_dict.get("moshi_name", "model.safetensors"))
        tok_path   = hf_hub_download(HF_REPO, cfg_dict["tokenizer_name"])

        lm_config = models.LmConfig.from_config_dict(cfg_dict)
        lm = models.Lm(lm_config)
        lm.set_dtype(mx.bfloat16)
        if model_path.endswith(".q8.safetensors"):
            nn.quantize(lm, bits=8, group_size=64)
        elif model_path.endswith(".q4.safetensors"):
            nn.quantize(lm, bits=4, group_size=32)
        lm.load_weights(model_path, strict=True)

        text_tok = sentencepiece.SentencePieceProcessor(tok_path)
        n_mimi = max(lm_config.generated_codebooks, lm_config.other_codebooks)
        audio_tok = rustymimi.Tokenizer(mimi_path, num_codebooks=n_mimi)
        ct = None
        lm.warmup(ct)
        print("Model loaded\n")
    except Exception as e:
        import traceback
        print(f"  MODEL LOAD FAILED: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return

    chunk_samples   = int(SAMPLE_RATE * CHUNK_SECONDS)
    n = 0

    max_buffer = chunk_samples * 3  # if more than ~9s buffered, we're behind

    while is_running:
        try:
            with audio_lock:
                blen = len(audio_buffer)
            with metrics_lock:
                metrics["audio_buffer_seconds"] = blen / SAMPLE_RATE
                metrics["transcript_queue_size"] = transcript_queue.qsize()
            if blen < chunk_samples:
                time.sleep(0.1)
                continue

            with audio_lock:
                # If buffer has grown too large, skip ahead to stay near real-time
                if len(audio_buffer) > max_buffer:
                    skip = len(audio_buffer) - chunk_samples
                    del audio_buffer[:skip]
                    print(f"  ⏩ Skipped {skip/SAMPLE_RATE:.1f}s of audio to catch up")
                    with metrics_lock:
                        metrics["chunks_skipped_catchup"] += 1
                chunk = audio_buffer[:chunk_samples]
                del audio_buffer[:chunk_samples]

            arr = np.array(chunk, dtype=np.float32)
            rms = np.sqrt(np.mean(arr ** 2))
            n += 1
            print(f"  Chunk #{n}: RMS={rms:.4f}", end=" ", flush=True)
            if rms < MIN_RMS:
                print("(silent, skipped)")
                with metrics_lock:
                    metrics["chunks_skipped_silent"] += 1
                continue

            print("→ running STT...")
            in_pcms = arr[np.newaxis, :]  # (1, samples)
            if stt_cfg is not None:
                pad_l = int(stt_cfg.get("audio_silence_prefix_seconds", 0.0) * SAMPLE_RATE)
                pad_r = int((stt_cfg.get("audio_delay_seconds", 0.0) + 1.0) * SAMPLE_RATE)
                in_pcms = np.pad(in_pcms, [(0, 0), (pad_l, pad_r)])

            # Reset model KV cache and recreate audio tokenizer before each chunk
            # (reset() alone doesn't fully clear rustymimi internal state)
            lm.warmup(ct)
            t_tok = time.time()
            audio_tok = rustymimi.Tokenizer(mimi_path, num_codebooks=n_mimi)
            with metrics_lock:
                metrics["tokenizer_recreations"] += 1

            steps = in_pcms.shape[-1] // 1920
            gen = models.LmGen(
                model=lm, max_steps=steps,
                text_sampler=utils.Sampler(top_k=25, temp=0.0),
                audio_sampler=utils.Sampler(top_k=250, temp=0.0),
                cfg_coef=1.0, check=False,
            )

            t0 = time.time()
            parts = []
            for i in range(steps):
                pcm = in_pcms[:, i * 1920:(i + 1) * 1920]
                tokens = audio_tok.encode_step(pcm[None, 0:1])
                tokens = mx.array(tokens).transpose(0, 2, 1)[:, :, :lm_config.other_codebooks]
                tok_id = gen.step(tokens[0], ct)[0].item()
                if tok_id not in (0, 3):
                    parts.append(text_tok.id_to_piece(tok_id).replace("▁", " "))

            result = "".join(parts).strip()
            dt = time.time() - t0
            print(f"  STT ({dt:.1f}s): '{result}'")

            with metrics_lock:
                metrics["chunks_processed"] += 1
                metrics["stt_last_duration"] = dt
                metrics["stt_total_time"] += dt
                metrics["stt_avg_duration"] = metrics["stt_total_time"] / metrics["chunks_processed"]
                metrics["stt_last_text"] = result
                if not result:
                    metrics["stt_empty_results"] += 1

            if result:
                print(f"  → sending: {result[:100]}")
                transcript_queue.put({
                    "type": "transcript",
                    "text": result,
                    "timestamp": time.time()
                })

        except Exception as e:
            import traceback
            print(f"  STT loop error (continuing): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            time.sleep(2)


async def _proxy_claude(websocket, req):
    """Proxy a Claude API request server-side so the key never touches the browser."""
    import aiohttp
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
                await websocket.send(json.dumps({
                    "type": "claude_response",
                    "status": resp.status,
                    "data": data,
                    "req_id": req.get("req_id"),
                }))
    except Exception as e:
        await websocket.send(json.dumps({
            "type": "claude_response",
            "status": 500,
            "data": {"error": str(e)},
            "req_id": req.get("req_id"),
        }))


async def ws_handler(websocket):
    connected_clients.add(websocket)
    with metrics_lock:
        metrics["ws_clients"] = len(connected_clients)
    print(f"Browser connected: {websocket.remote_address}")
    await websocket.send(json.dumps({
        "type": "status", "status": "connected",
        "message": "STT server ready"
    }))
    try:
        async for msg in websocket:
            d = json.loads(msg)
            if d.get("type") == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            elif d.get("type") == "get_metrics":
                with metrics_lock:
                    m = {**metrics, "uptime": time.time() - metrics["started_at"]}
                await websocket.send(json.dumps({"type": "metrics", **m}))
            elif d.get("type") == "claude_request":
                asyncio.create_task(_proxy_claude(websocket, d))
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)
        with metrics_lock:
            metrics["ws_clients"] = len(connected_clients)
        print(f"Browser disconnected: {websocket.remote_address}")


async def broadcast():
    while is_running:
        try:
            msg = transcript_queue.get_nowait()
            if connected_clients:
                p = json.dumps(msg)
                await asyncio.gather(
                    *[c.send(p) for c in connected_clients],
                    return_exceptions=True)
        except queue.Empty:
            pass
        await asyncio.sleep(0.05)


async def main(device_idx):
    threading.Thread(target=audio_capture_thread, args=(device_idx,), daemon=True).start()
    threading.Thread(target=stt_thread, daemon=True).start()
    import websockets
    print(f"WebSocket: ws://{WS_HOST}:{WS_PORT}")
    print("Open live-mindmap.html in Chrome\n")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await broadcast()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live Mind Map STT Server")
    p.add_argument("-d", "--device", type=int, default=None,
                   help="Input device index (skips interactive picker)")
    args = p.parse_args()

    print("=" * 50)
    print("  Live Mind Map : STT Server")
    print("=" * 50 + "\n")
    device_idx = select_input_device(forced=args.device)

    try:
        asyncio.run(main(device_idx))
    except KeyboardInterrupt:
        print("\nDone.")
        is_running = False