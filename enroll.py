"""Local speaker enrollment CLI.

Records voice samples via microphone and creates a speaker profile (.npy).
Copy the resulting profile to the transcriber's profiles/ directory.

Usage:
    python enroll.py --device-id desktop-listener --output ./profiles/desktop-listener.npy
    python enroll.py --device-id desktop-listener --samples 3
"""

import argparse
import io
import logging
import struct
import sys
import wave

import numpy as np

from speaker_gate import SpeakerGate

log = logging.getLogger("enroll")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def record_sample(index, total, duration_s=5.0):
    """Record a single voice sample from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        log.error("sounddevice is required for enrollment: pip install sounddevice")
        sys.exit(1)

    print(f"\n--- Sample {index}/{total} ---")
    print(f"Speak naturally for up to {duration_s:.0f} seconds. Press Ctrl+C to stop early.")
    input("Press Enter when ready to record...")

    frames = int(duration_s * SAMPLE_RATE)
    print("Recording...")
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    sd.wait()
    print("Done.")

    audio = audio.flatten()
    # Trim trailing silence (below -40dB)
    threshold = 0.01
    nonsilent = np.where(np.abs(audio) > threshold)[0]
    if len(nonsilent) > 0:
        audio = audio[:nonsilent[-1] + int(0.5 * SAMPLE_RATE)]  # keep 0.5s after last speech

    duration = len(audio) / SAMPLE_RATE
    print(f"Captured {duration:.1f}s of audio")
    return audio


def main():
    parser = argparse.ArgumentParser(description="Enroll speaker voice profile locally")
    parser.add_argument("--device-id", required=True, help="Device/user identifier for the profile")
    parser.add_argument("--output", default=None, help="Output path for .npy profile (default: ./profiles/{device-id}.npy)")
    parser.add_argument("--samples", type=int, default=3, help="Number of voice samples to record (default: 3)")
    parser.add_argument("--duration", type=float, default=5.0, help="Max seconds per sample (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.45, help="Verification threshold (default: 0.45)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = args.output
    if not output_path:
        output_path = f"./profiles/{args.device_id}.npy"

    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Use a temporary profiles dir, then move the result
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    gate = SpeakerGate(tmp_dir, args.threshold)

    print("=== Speaker Enrollment ===")
    print(f"Device ID: {args.device_id}")
    print(f"Samples to record: {args.samples}")
    print(f"Output: {output_path}")
    print()
    print("Loading speaker verification model...")
    gate.load_model()
    print("Model loaded.")

    audio_samples = []
    for i in range(1, args.samples + 1):
        audio = record_sample(i, args.samples, args.duration)
        audio_samples.append(audio)

    print("\nComputing speaker embedding...")
    info = gate.enroll(args.device_id, audio_samples)
    print(f"Embedding dimension: {info['embedding_dim']}")
    print(f"Samples used: {info['samples_used']}")

    # Move the generated profile to the desired output path
    src = os.path.join(tmp_dir, f"{args.device_id}.npy")
    import shutil
    shutil.move(src, output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nProfile saved to: {output_path}")
    print("Copy this file to the transcriber's profiles/ directory on the server.")


if __name__ == "__main__":
    main()
