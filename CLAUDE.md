# CLAUDE.md -- Transcriber Service

This file provides guidance to Claude Code when working in this repository.

IMPORTANT: NEVER USE EMOJIS ANYWHERE IN LOGGING, CODE OR OTHER TEXT.

IMPORTANT: Treat this codebase as work in progress. Never do backwards-compatibility or legacy support unless explicitly asked. Remove code that becomes redundant.

## What this service is

An audio transcription microservice built on faster-whisper, behind a FastAPI HTTP +
WebSocket API. It supports batch transcription, real-time streaming with VAD, optional
per-device speaker gating (SpeechBrain ECAPA-TDNN), and an optional orchestrator bridge and
inline translation. It is the transcription backend for the AI orchestration system.

This service listens on **port 10003**.

Status: built but NOT currently deployed (there is no `.gitlab-ci.yml`; it is not part of the
auto-deploy pipeline). If/when it is wired up for deployment, it follows the same CI pattern
as the other services (see Deploy below).

For the whole-system architecture and the full port table, see the orchestrator repo
(`jaskier-os/orchestrator`) CLAUDE.md / docs -- do not duplicate that map here.

## Tech stack

- Python 3 + FastAPI (HTTP + WebSocket)
- faster-whisper / CTranslate2 (CPU or CUDA via `DEVICE`, `COMPUTE_TYPE`)
- Optional: SpeechBrain ECAPA-TDNN speaker verification, Silero VAD

## Run

```bash
bash run.sh   # venv + requirements + sets LD_LIBRARY_PATH for cuBLAS/cuDNN, runs main.py
```

Copy `.env.example` to `.env`. Key knobs: `WHISPER_MODEL`, `DEVICE`, `COMPUTE_TYPE`,
`BEAM_SIZE`, `LANGUAGE`, `PORT`, `API_KEY` (required). Speaker gating: `SPEAKER_ENABLED`,
`SPEAKER_THRESHOLD`, `SPEAKER_PROFILES_DIR` (enroll with `enroll.py`). Streaming/VAD:
`STREAMING_*`. Integrations: `TRANSLATOR_URL` (default the translator service on 10015),
`ORCHESTRATOR_WS_URL` (empty disables the bridge), `AUDIO_DUMP_PATH`, `TLS_CERT_PATH`.

## Layout

- `main.py` -- FastAPI app: `GET /health`, `POST /transcribe`, `WS /ws/stream`
- `config.py` -- env-driven configuration
- `streaming.py` -- streaming + VAD logic
- `speaker_gate.py` -- speaker verification gating
- `orchestrator_bridge.py` -- outbound orchestrator WebSocket bridge
- `enroll.py` -- build speaker profiles ({device_id}.npy in `profiles/`)
- `export_onnx.py` -- ONNX export helper
- `tests/`, `test_e2e.py` -- tests

## API

- `GET /health` -- no auth
- `POST /transcribe` -- batch; auth via `Authorization: Bearer <API_KEY>`
- `WS /ws/stream` -- streaming; auth via `x-api-key`

## Deploy

If deployment is enabled, this is a server-side service that auto-deploys on git push to
`main`: CI builds the Dockerfile with `docker buildx --push` to the local registry, then
bumps the image tag in the `infrastructure/deploy` repo, which Flux reconciles onto the
cluster.

- NEVER restart the running service yourself -- if a restart is needed, tell the user.
- NEVER build/push Docker images by hand; commit + push and let CI handle it.
- NEVER modify Kubernetes env/secrets/deployments directly; changes go through the deploy
  manifests repo so Flux reconciles them.
