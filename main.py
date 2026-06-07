import asyncio
import io
import json
import os
import wave
import struct
import logging
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, Header, HTTPException, WebSocket, WebSocketDisconnect

from config import (
    WHISPER_MODEL, DEVICE, COMPUTE_TYPE, PORT, API_KEY, BEAM_SIZE, LANGUAGE, CPU_THREADS,
    SPEAKER_ENABLED, SPEAKER_THRESHOLD, SPEAKER_PROFILES_DIR,
    ORCHESTRATOR_WS_URL,
    TRANSLATOR_URL, STREAMING_VAD_THRESHOLD, STREAMING_MIN_CHUNK_SIZE, STREAMING_SILENCE_THRESHOLD,
    STREAMING_SENTENCE_SILENCE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transcriber")

model = None
speaker_gate = None
vad_ort_session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, speaker_gate, vad_ort_session
    from faster_whisper import WhisperModel

    log.info("Loading whisper model '%s' on %s (%s)...", WHISPER_MODEL, DEVICE, COMPUTE_TYPE)
    model = WhisperModel(
        WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        num_workers=6,
    )
    log.info("Whisper model loaded.")

    if SPEAKER_ENABLED:
        from speaker_gate import SpeakerGate
        log.info("Loading speaker verification model (threshold=%.2f, profiles=%s)...",
                 SPEAKER_THRESHOLD, SPEAKER_PROFILES_DIR)
        speaker_gate = SpeakerGate(SPEAKER_PROFILES_DIR, SPEAKER_THRESHOLD)
        speaker_gate.load_model()
        log.info("Speaker verification model loaded.")

    # Load Silero VAD ONNX
    import onnxruntime
    vad_model_path = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")
    if not os.path.exists(vad_model_path):
        import urllib.request
        log.info("Downloading Silero VAD ONNX model...")
        urllib.request.urlretrieve(
            "https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx",
            vad_model_path,
        )
    vad_ort_session = onnxruntime.InferenceSession(vad_model_path)
    log.info("Silero VAD ONNX model loaded.")

    # Connect to orchestrator if URL is configured
    bridge = None
    if ORCHESTRATOR_WS_URL:
        from orchestrator_bridge import OrchestratorBridge
        bridge = OrchestratorBridge(
            orchestrator_url=ORCHESTRATOR_WS_URL,
            api_key=API_KEY,
            whisper_model=model,
            vad_session=vad_ort_session,
            streaming_config={
                "min_chunk_size": STREAMING_MIN_CHUNK_SIZE,
                "silence_threshold": STREAMING_SILENCE_THRESHOLD,
                "sentence_silence": STREAMING_SENTENCE_SILENCE,
                "vad_threshold": STREAMING_VAD_THRESHOLD,
                "beam_size": BEAM_SIZE,
                "translator_url": TRANSLATOR_URL,
                "api_key": API_KEY,
                "audio_dump_path": os.environ.get("AUDIO_DUMP_PATH", ""),
            },
        )
        await bridge.start()
        log.info("Orchestrator bridge started: %s", ORCHESTRATOR_WS_URL)

    yield

    if bridge:
        await bridge.stop()
    log.info("Shutting down.")


app = FastAPI(lifespan=lifespan)


def verify_auth(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(data)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = struct.unpack(fmt, raw)
        audio = np.array(samples, dtype=np.float32) / 32768.0
    elif sample_width == 4:
        fmt = f"<{n_frames * n_channels}i"
        samples = struct.unpack(fmt, raw)
        audio = np.array(samples, dtype=np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sample_rate


def decode_raw_pcm(data: bytes, sample_rate: int) -> tuple[np.ndarray, int]:
    n_samples = len(data) // 2
    fmt = f"<{n_samples}h"
    samples = struct.unpack(fmt, data)
    audio = np.array(samples, dtype=np.float32) / 32768.0
    return audio, sample_rate


def decode_audio(data: bytes, sample_rate: int) -> tuple[np.ndarray, int]:
    """Decode audio data (WAV or raw PCM) into float32 samples."""
    if data[:4] == b"RIFF":
        return decode_wav(data)
    return decode_raw_pcm(data, sample_rate)


@app.get("/health")
async def health():
    return {"status": "ok", "speaker_enabled": SPEAKER_ENABLED}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    sample_rate: int = Query(16000),
    device_id: str = Query(""),
    verify_speaker: bool = Query(False),
    language: str | None = Query(None),
    authorization: str | None = Header(None),
):
    verify_auth(authorization)

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    log.info("Received audio: %d bytes, content_type=%s", len(data), audio.content_type)

    try:
        samples, sr = decode_audio(data, sample_rate)
    except Exception as e:
        log.error("Failed to decode audio: %s", e)
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}")

    duration = len(samples) / sr

    # Speaker verification (if requested and gate is available)
    speaker_verified = None
    speaker_similarity = None
    if verify_speaker and speaker_gate:
        verified, similarity = speaker_gate.verify(device_id, samples)
        speaker_verified = verified
        speaker_similarity = round(similarity, 4)
        log.info("Speaker verification: verified=%s, similarity=%.4f", verified, similarity)

    log.info("Transcribing %.2fs of audio (sample_rate=%d)...", duration, sr)

    try:
        effective_language = language or LANGUAGE
        segments, info = model.transcribe(
            samples,
            beam_size=BEAM_SIZE,
            word_timestamps=False,
            language=effective_language,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())
        text = " ".join(texts)
        detected_language = info.language if info else ""
    except Exception as e:
        log.error("Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail="Transcription failed")

    log.info("Transcription complete: language=%s, text_length=%d", detected_language, len(text))

    result = {"text": text, "language": detected_language, "duration": round(duration, 2)}
    if speaker_verified is not None:
        result["speaker_verified"] = speaker_verified
        result["speaker_similarity"] = speaker_similarity
    return result


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    from streaming import StreamingSession

    await websocket.accept()
    session = None
    try:
        start_msg = await websocket.receive_text()
        data = json.loads(start_msg)
        if data.get("type") != "start":
            await websocket.close(code=4400, reason="Expected start message")
            return

        config = {
            "sample_rate": data.get("sampleRate", 16000),
            "source_lang": data.get("sourceLang") or None,
            "target_lang": data.get("targetLang", "ru"),
            "source_nllb": data.get("sourceNllb", "eng_Latn"),
            "target_nllb": data.get("targetNllb", "rus_Cyrl"),
            "translate": data.get("translate", False),
            "min_chunk_size": STREAMING_MIN_CHUNK_SIZE,
            "silence_threshold": STREAMING_SILENCE_THRESHOLD,
            "vad_threshold": STREAMING_VAD_THRESHOLD,
            "translator_url": TRANSLATOR_URL,
            "api_key": API_KEY,
        }

        session = StreamingSession(
            ws=websocket,
            whisper_model=model,
            config=config,
            vad_session=vad_ort_session,
        )
        await session.start()

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message:
                session.feed_audio(message["bytes"])
            elif "text" in message:
                text_data = json.loads(message["text"])
                msg_type = text_data.get("type")
                if msg_type == "stop":
                    break
                # speech_start/speech_end ignored -- session auto-starts,
                # server handles silence via RMS threshold
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("Streaming session error: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        if session:
            await session.stop()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
