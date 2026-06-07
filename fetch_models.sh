#!/usr/bin/env bash
# Download standard pretrained models the transcriber needs at runtime.
#
# - Silero VAD ONNX: voice-activity detection model used for streaming.
# - faster-whisper weights: downloaded automatically from Hugging Face on first
#   service start, based on WHISPER_MODEL. No action needed here.
# - SpeechBrain ECAPA-TDNN (speaker verification): downloaded automatically from
#   Hugging Face on first use when SPEAKER_ENABLED=true. No action needed here.
#
# These are all standard, publicly downloadable models and are intentionally not
# committed to the repository.
set -euo pipefail
cd "$(dirname "$0")"

VAD_URL="https://github.com/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx"
VAD_OUT="silero_vad.onnx"

if [ -f "$VAD_OUT" ]; then
  echo "[fetch_models] $VAD_OUT already present, skipping."
else
  echo "[fetch_models] Downloading Silero VAD ONNX..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$VAD_URL" -o "$VAD_OUT"
  else
    wget -q "$VAD_URL" -O "$VAD_OUT"
  fi
  echo "[fetch_models] Saved $VAD_OUT."
fi

echo "[fetch_models] Done. Whisper and SpeechBrain weights download on first run."
