"""
STT Worker — Audio capture + VAD + Kyutai STT inference.
Extracted from stt_server.py. Runs as background threads, pushes
transcript messages into a shared queue consumed by app.py.
"""

import time, threading, queue, sys
import numpy as np

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SECONDS = 2
MIN_RMS = 0.01


class VAD:
    """Energy-based voice activity detector with hysteresis."""

    def __init__(self, alpha=0.3, onset_threshold=0.02, offset_threshold=0.008,
                 onset_frames=3, offset_frames=15):
        self.alpha = alpha
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.onset_frames = onset_frames      # ~90ms at 30ms blocks
        self.offset_frames = offset_frames    # ~450ms
        self.smoothed_rms = 0.0
        self.is_speaking = False
        self._onset_count = 0
        self._offset_count = 0

    def process_frame(self, rms: float) -> str | None:
        """Update with a new frame RMS. Returns 'speech_start', 'speech_end', or None."""
        self.smoothed_rms = self.alpha * rms + (1 - self.alpha) * self.smoothed_rms

        if not self.is_speaking:
            if self.smoothed_rms > self.onset_threshold:
                self._onset_count += 1
                if self._onset_count >= self.onset_frames:
                    self.is_speaking = True
                    self._onset_count = 0
                    self._offset_count = 0
                    return "speech_start"
            else:
                self._onset_count = 0
        else:
            if self.smoothed_rms < self.offset_threshold:
                self._offset_count += 1
                if self._offset_count >= self.offset_frames:
                    self.is_speaking = False
                    self._offset_count = 0
                    self._onset_count = 0
                    return "speech_end"
            else:
                self._offset_count = 0

        return None


def select_input_device(forced=None):
    """Interactive mic picker. Returns device index."""
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


def _audio_capture_thread(device_idx, audio_buffer, audio_lock, vad, metrics, metrics_lock, shutdown):
    """Capture mic audio, resample to 24kHz, feed VAD."""
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

            # VAD
            event = vad.process_frame(rms)
            if event:
                with metrics_lock:
                    metrics["vad_state"] = "speaking" if event == "speech_start" else "silent"

            now = time.time()
            if now - rms_report["last"] > 5.0:
                vad_label = "speaking" if vad.is_speaking else "silent"
                print(f"  Audio RMS: {rms:.4f} ({vad_label})")
                rms_report["last"] = now
            with audio_lock:
                audio_buffer.extend(data.tolist())

        with sd.InputStream(samplerate=native_rate, channels=CHANNELS,
                            dtype="float32", callback=cb, device=device_idx,
                            blocksize=int(native_rate * 0.03)):
            while not shutdown.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"  AUDIO CAPTURE FAILED: {e}", file=sys.stderr)
        print("  Check: System Settings > Privacy & Security > Microphone > Terminal", file=sys.stderr)


def _stt_thread(audio_buffer, audio_lock, vad, transcript_queue, metrics, metrics_lock, shutdown):
    """Load Kyutai STT model and transcribe chunks from audio_buffer."""
    print("Loading Kyutai STT model...")
    try:
        import json as _json
        import mlx.core as mx
        import mlx.nn as nn
        import rustymimi
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi_mlx import models, utils

        HF_REPO = "kyutai/stt-1b-en_fr-mlx"
        cfg_path = hf_hub_download(HF_REPO, "config.json")
        with open(cfg_path) as f:
            cfg_dict = _json.load(f)

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

    chunk_samples = int(SAMPLE_RATE * CHUNK_SECONDS)
    max_buffer = chunk_samples * 3
    n = 0

    while not shutdown.is_set():
        try:
            with audio_lock:
                blen = len(audio_buffer)
            with metrics_lock:
                metrics["audio_buffer_seconds"] = blen / SAMPLE_RATE
                metrics["transcript_queue_size"] = transcript_queue.qsize()
            if blen < chunk_samples:
                time.sleep(0.1)
                continue

            # VAD-aware silence skip: if VAD says silent AND buffer isn't overflowing, skip
            if not vad.is_speaking and blen < max_buffer:
                with audio_lock:
                    del audio_buffer[:chunk_samples]
                with metrics_lock:
                    metrics["chunks_skipped_silent"] += 1
                continue

            with audio_lock:
                if len(audio_buffer) > max_buffer:
                    skip = len(audio_buffer) - chunk_samples
                    del audio_buffer[:skip]
                    print(f"  >>> Skipped {skip/SAMPLE_RATE:.1f}s of audio to catch up")
                    with metrics_lock:
                        metrics["chunks_skipped_catchup"] += 1
                chunk = audio_buffer[:chunk_samples]
                del audio_buffer[:chunk_samples]

            t_e2e = time.time()
            arr = np.array(chunk, dtype=np.float32)
            rms = np.sqrt(np.mean(arr ** 2))
            n += 1
            print(f"  Chunk #{n}: RMS={rms:.4f}", end=" ", flush=True)
            if rms < MIN_RMS:
                print("(silent, skipped)")
                with metrics_lock:
                    metrics["chunks_skipped_silent"] += 1
                continue

            print("-> running STT...")
            in_pcms = arr[np.newaxis, :]
            if stt_cfg is not None:
                pad_l = int(stt_cfg.get("audio_silence_prefix_seconds", 0.0) * SAMPLE_RATE)
                pad_r = int((stt_cfg.get("audio_delay_seconds", 0.0) + 1.0) * SAMPLE_RATE)
                in_pcms = np.pad(in_pcms, [(0, 0), (pad_l, pad_r)])

            lm.warmup(ct)
            t_tok = time.time()
            audio_tok = rustymimi.Tokenizer(mimi_path, num_codebooks=n_mimi)
            tok_ms = (time.time() - t_tok) * 1000
            with metrics_lock:
                metrics["tokenizer_recreations"] += 1
                metrics["tokenizer_last_ms"] = tok_ms

            steps = in_pcms.shape[-1] // 1920
            gen = models.LmGen(
                model=lm, max_steps=steps,
                text_sampler=utils.Sampler(top_k=25, temp=0.0),
                audio_sampler=utils.Sampler(top_k=250, temp=0.0),
                cfg_coef=1.0, check=False,
            )

            PARTIAL_INTERVAL = 4

            t0 = time.time()
            parts = []
            for i in range(steps):
                pcm = in_pcms[:, i * 1920:(i + 1) * 1920]
                tokens = audio_tok.encode_step(pcm[None, 0:1])
                tokens = mx.array(tokens).transpose(0, 2, 1)[:, :, :lm_config.other_codebooks]
                tok_id = gen.step(tokens[0], ct)[0].item()
                if tok_id not in (0, 3):
                    parts.append(text_tok.id_to_piece(tok_id).replace("\u2581", " "))

                if (i + 1) % PARTIAL_INTERVAL == 0 and parts:
                    partial_text = "".join(parts).strip()
                    if partial_text:
                        transcript_queue.put({
                            "type": "partial_transcript",
                            "text": partial_text,
                            "timestamp": time.time(),
                        })
                        with metrics_lock:
                            metrics["stt_partials_emitted"] += 1

            result = "".join(parts).strip()
            dt = time.time() - t0
            print(f"  STT ({dt:.1f}s): '{result}'")

            e2e = time.time() - t_e2e
            with metrics_lock:
                metrics["chunks_processed"] += 1
                metrics["stt_last_duration"] = dt
                metrics["stt_total_time"] += dt
                metrics["stt_avg_duration"] = metrics["stt_total_time"] / metrics["chunks_processed"]
                metrics["stt_last_text"] = result
                metrics["stt_e2e_last"] = e2e
                metrics["stt_e2e_total"] += e2e
                metrics["stt_e2e_avg"] = metrics["stt_e2e_total"] / metrics["chunks_processed"]
                if not result:
                    metrics["stt_empty_results"] += 1

            if result:
                print(f"  -> sending: {result[:100]}")
                transcript_queue.put({
                    "type": "transcript",
                    "text": result,
                    "timestamp": time.time(),
                })

        except Exception as e:
            import traceback
            print(f"  STT loop error (continuing): {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            time.sleep(2)


def start_stt_pipeline(device_idx, transcript_queue, metrics, metrics_lock):
    """Spawn audio capture + STT threads. Returns (shutdown_event, vad)."""
    shutdown = threading.Event()
    audio_buffer = []
    audio_lock = threading.Lock()
    vad = VAD()

    t1 = threading.Thread(
        target=_audio_capture_thread,
        args=(device_idx, audio_buffer, audio_lock, vad, metrics, metrics_lock, shutdown),
        daemon=True,
    )
    t2 = threading.Thread(
        target=_stt_thread,
        args=(audio_buffer, audio_lock, vad, transcript_queue, metrics, metrics_lock, shutdown),
        daemon=True,
    )
    t1.start()
    t2.start()
    return shutdown, vad
