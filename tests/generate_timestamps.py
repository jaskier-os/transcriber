"""Generate sentence-level timestamps from test audio using faster-whisper.
Run once to create audio_timestamps.json reference file.

Usage: python tests/generate_timestamps.py
"""
import json
import os

import numpy as np
from faster_whisper import WhisperModel

PCM_FILE = os.path.join(os.path.dirname(__file__), "test_30s.pcm")
OUTPUT = os.path.join(os.path.dirname(__file__), "audio_timestamps.json")

pcm = np.fromfile(PCM_FILE, dtype=np.int16).astype(np.float32) / 32768.0
duration = len(pcm) / 16000
print(f"Audio: {duration:.1f}s, {len(pcm)} samples")

print("Loading model...")
model = WhisperModel("large-v3", device="cuda", compute_type="int8")

print("Transcribing with word timestamps...")
segments, info = model.transcribe(
    pcm, language="en", word_timestamps=True, beam_size=5,
    condition_on_previous_text=True, vad_filter=True,
)

sentences = []
words = []
for seg in segments:
    sentences.append({
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip(),
    })
    for w in seg.words:
        words.append({
            "start": round(w.start, 2),
            "end": round(w.end, 2),
            "word": w.word.strip(),
        })

with open(OUTPUT, "w") as f:
    json.dump({"sentences": sentences, "words": words}, f, indent=2)

print(f"\nSaved {len(sentences)} sentences, {len(words)} words to {OUTPUT}")
print(f"\nSentence timestamps:")
for s in sentences:
    print(f"  [{s['start']:5.1f}s - {s['end']:5.1f}s] {s['text']}")
