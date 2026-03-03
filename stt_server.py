#!/usr/bin/env python3
"""
Live Mind Map - STT Server
Captures mic, transcribes with Kyutai STT MLX, streams to browser via WebSocket.
"""

import asyncio, json, time, threading, queue, sys
import numpy as np

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SECONDS = 8
OVERLAP_SECONDS = 1
WS_HOST = "localhost"
WS_PORT = 8765
MIN_RMS = 0.01

transcript_queue = queue.Queue()
audio_buffer = []
audio_lock = threading.Lock()
is_running = True
connected_clients = set()


def audio_capture_thread():
    import sounddevice as sd
    try:
        dev = sd.query_devices(kind="input")
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
            now = time.time()
            if now - rms_report["last"] > 5.0:
                print(f"  Audio RMS: {rms:.4f} ({'OK' if rms > MIN_RMS else 'silent/quiet'})")
                rms_report["last"] = now
            with audio_lock:
                audio_buffer.extend(data.tolist())

        with sd.InputStream(samplerate=native_rate, channels=CHANNELS,
                            dtype="float32", callback=cb,
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
    overlap_samples = int(SAMPLE_RATE * OVERLAP_SECONDS)
    last_text = ""
    n = 0

    while is_running:
        time.sleep(1)
        with audio_lock:
            blen = len(audio_buffer)
        if blen < chunk_samples:
            print(f"  STT waiting: {blen}/{chunk_samples} samples buffered")
            continue

        with audio_lock:
            chunk = audio_buffer[:chunk_samples]
            del audio_buffer[:chunk_samples - overlap_samples]

        arr = np.array(chunk, dtype=np.float32)
        rms = np.sqrt(np.mean(arr ** 2))
        n += 1
        print(f"  Chunk #{n}: RMS={rms:.4f}", end=" ", flush=True)
        if rms < MIN_RMS:
            print("(silent, skipped)")
            continue

        print("→ running STT...")
        try:
            in_pcms = arr[np.newaxis, :]  # (1, samples)
            if stt_cfg is not None:
                pad_l = int(stt_cfg.get("audio_silence_prefix_seconds", 0.0) * SAMPLE_RATE)
                pad_r = int((stt_cfg.get("audio_delay_seconds", 0.0) + 1.0) * SAMPLE_RATE)
                in_pcms = np.pad(in_pcms, [(0, 0), (pad_l, pad_r)])

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

            if result:
                new = result
                if last_text:
                    ov = _overlap(last_text, new)
                    if ov:
                        new = new[len(ov):].strip()
                if new:
                    print(f"  → sending: {new[:100]}")
                    transcript_queue.put({
                        "type": "transcript",
                        "text": new,
                        "timestamp": time.time()
                    })
                    last_text = result
        except Exception as e:
            import traceback
            print(f"  STT error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def _overlap(prev, curr):
    pw, cw = prev.split(), curr.split()
    best = ""
    for n in range(1, min(len(pw), len(cw), 10) + 1):
        if " ".join(pw[-n:]).lower() == " ".join(cw[:n]).lower():
            best = " ".join(cw[:n])
    return best


async def ws_handler(websocket):
    connected_clients.add(websocket)
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
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)
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


async def main():
    print("=" * 50)
    print("  Live Mind Map : STT Server")
    print("=" * 50 + "\n")
    threading.Thread(target=audio_capture_thread, daemon=True).start()
    threading.Thread(target=stt_thread, daemon=True).start()
    import websockets
    print(f"WebSocket: ws://{WS_HOST}:{WS_PORT}")
    print("Open live-mindmap.html in Chrome\n")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await broadcast()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDone.")
        is_running = False