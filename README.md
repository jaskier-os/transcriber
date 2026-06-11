# transcriber

> **Docs & wiki:** [github.com/jaskier-os/docs/wiki](https://github.com/jaskier-os/docs/wiki)

A FastAPI speech-to-text service built on faster-whisper. `POST /transcribe`
handles one-shot WAV/PCM clips; `WS /ws/stream` does low-latency streaming
transcription (and optional inline translation) using Silero VAD for speech
boundaries. Optional per-device speaker gating (SpeechBrain ECAPA-TDNN) and an
optional outbound orchestrator bridge. Built but not currently deployed.

## Build / run

```bash
cp .env.example .env   # then edit it
./fetch_models.sh      # fetches the Silero VAD model
./run.sh               # creates venv, installs deps, wires CUDA libs, starts
```

To run manually instead of `run.sh`:

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python main.py
```

Listens on `0.0.0.0:$PORT` (default 10003); check `GET /health`. Docker (CPU
image, downloads models at build time):

```bash
docker build -t transcriber .
docker run --rm -p 10003:10003 --env-file .env transcriber
```

## Configuration

Config lives in environment variables. `.env.example` is the source of truth -
copy it to `.env` and edit; each var is documented inline.

Set at least `API_KEY` (bearer token clients must present). Others worth knowing:
`WHISPER_MODEL` (e.g. `large-v3`, `medium`), `DEVICE` / `COMPUTE_TYPE` (`cpu`/`cuda`
and the matching compute type), `PORT` (default 10003), `SPEAKER_ENABLED`, and the
optional `TRANSLATOR_URL` / `ORCHESTRATOR_WS_URL` integrations (leave the
orchestrator URL empty to disable the bridge).

## Dependencies

Python 3.12 (Docker uses `python:3.12-slim`). Key packages: `fastapi`, `uvicorn`,
`faster-whisper`, `speechbrain`, `onnxruntime`, `numpy`. The Docker build installs
CPU `torch` from the PyTorch index.

Models are not committed: Silero VAD comes from `fetch_models.sh`; whisper weights
download from Hugging Face on first start (per `WHISPER_MODEL`); the SpeechBrain
ECAPA model downloads on first use when speaker gating is enabled.
