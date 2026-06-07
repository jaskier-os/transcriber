# Transcriber

A speech-to-text service built on faster-whisper. It exposes an HTTP endpoint for
one-shot transcription and a WebSocket endpoint for low-latency streaming
transcription with optional inline translation. Streaming uses Silero VAD for
speech-boundary detection. An optional speaker-verification gate (SpeechBrain
ECAPA-TDNN) can restrict transcription to enrolled speakers, and an optional
outbound bridge lets an orchestrator relay audio to the service.

## Features

- `POST /transcribe` - transcribe a WAV or raw 16-bit PCM clip.
- `WS /ws/stream` - streaming transcription (and optional translation) with VAD.
- `GET /health` - liveness probe.
- Optional per-device speaker verification.
- Optional orchestrator WebSocket bridge.

## Prerequisites

- Python 3.12
- For GPU inference: an NVIDIA GPU with CUDA-capable drivers (optional; the
  service runs on CPU by default).
- `curl` or `wget` (to fetch the VAD model).

## Setup

1. Copy the example environment file and fill in values:

   ```bash
   cp .env.example .env
   ```

   At minimum set `API_KEY`. Every variable is documented inline in
   `.env.example`; the important ones:

   - `API_KEY` - bearer token clients must present. Required.
   - `WHISPER_MODEL` - faster-whisper model name (e.g. `large-v3`, `medium`).
   - `DEVICE` / `COMPUTE_TYPE` - `cpu`/`cuda` and the matching compute type.
   - `PORT` - listen port (default `10003`).
   - `SPEAKER_ENABLED` - turn on speaker verification.
   - `TRANSLATOR_URL` / `ORCHESTRATOR_WS_URL` - optional integrations; leave the
     orchestrator URL empty to disable the bridge.

2. Fetch the Silero VAD model (whisper and speaker weights download on first run):

   ```bash
   ./fetch_models.sh
   ```

## Build / install

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
./run.sh
```

`run.sh` creates the venv if needed, installs dependencies, wires up CUDA libs if
present, and starts the service. To run manually:

```bash
venv/bin/python main.py
```

The service listens on `0.0.0.0:$PORT`. Check it:

```bash
curl http://localhost:10003/health
```

### Docker

```bash
docker build -t transcriber .
docker run --rm -p 10003:10003 --env-file .env transcriber
```

## Test

End-to-end tests assume a running instance and a valid `API_KEY`:

```bash
API_KEY=... TEST_URL=http://localhost:10003 venv/bin/python -m pytest test_e2e.py
```

Streaming tests live in `tests/test_streaming.py` and read `DIRECT_URL` /
`ORCHESTRATOR_URL` / `API_KEY` from the environment.

## Speaker enrollment (optional)

With `SPEAKER_ENABLED=true`, profiles are read from `SPEAKER_PROFILES_DIR` as
`{device_id}.npy`. Create one locally with a microphone:

```bash
venv/bin/python enroll.py --device-id my-device --samples 3
```

Copy the generated `.npy` into the server's `profiles/` directory. Profiles are
user data and are gitignored.

## Optional TLS / VPN

The service needs no certificate and no VPN to run. Outbound connections to the
translator and orchestrator use plain `http`/`ws` by default, and `https`/`wss`
URLs are verified against the system CA store. If your orchestrator/translator
uses a private CA, point `TLS_CERT_PATH` at a PEM bundle and it is added as an
extra trust anchor; if the variable is unset or the file is missing, the service
falls back to system CAs instead of failing. Certificate and key files are never
committed.

## Models

All models are standard, publicly downloadable, and not committed to the repo:

- **Silero VAD** (`silero_vad.onnx`) - fetched by `fetch_models.sh`.
- **faster-whisper** weights - downloaded from Hugging Face on first start based
  on `WHISPER_MODEL`.
- **SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`) - downloaded
  from Hugging Face on first use when speaker verification is enabled.
