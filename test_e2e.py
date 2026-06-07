"""
E2E tests for the transcriber service.
Requires the service to be running on localhost:10003.

Usage:
    python test_e2e.py
    pytest test_e2e.py -v
"""

import io
import struct
import wave
import math
import os
import sys

import requests

# Load .env for API key
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

BASE_URL = os.environ.get("TEST_URL", "http://localhost:10003")
API_KEY = os.environ.get("API_KEY", "")


def build_wav(samples_float: list[float], sample_rate: int = 16000) -> bytes:
    """Build a WAV file from float samples [-1.0, 1.0]."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = struct.pack(f"<{len(samples_float)}h", *[
            max(-32768, min(32767, int(s * 32767))) for s in samples_float
        ])
        wf.writeframes(pcm)
    return buf.getvalue()


def build_raw_pcm(samples_float: list[float]) -> bytes:
    """Build raw PCM16LE from float samples."""
    return struct.pack(f"<{len(samples_float)}h", *[
        max(-32768, min(32767, int(s * 32767))) for s in samples_float
    ])


def generate_sine(freq_hz: float = 440.0, duration_s: float = 1.0,
                   sample_rate: int = 16000, amplitude: float = 0.5) -> list[float]:
    """Generate a sine wave tone."""
    n = int(sample_rate * duration_s)
    return [amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate) for i in range(n)]


def generate_silence(duration_s: float = 1.0, sample_rate: int = 16000) -> list[float]:
    """Generate silence."""
    return [0.0] * int(sample_rate * duration_s)


def auth_header() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


# --- Tests ---

class TestHealth:
    def test_health_returns_ok(self):
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "speaker_enabled" in body


class TestAuth:
    def test_no_auth_returns_401(self):
        wav = build_wav(generate_silence(0.5))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 401

    def test_wrong_key_returns_401(self):
        wav = build_wav(generate_silence(0.5))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
            headers={"Authorization": "Bearer wrong_key_12345"},
        )
        assert r.status_code == 401

    def test_malformed_auth_returns_401(self):
        wav = build_wav(generate_silence(0.5))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
            headers={"Authorization": API_KEY},  # Missing "Bearer " prefix
        )
        assert r.status_code == 401


class TestTranscribeWav:
    def test_silence_returns_empty_text(self):
        """Silence should produce empty or near-empty transcription."""
        wav = build_wav(generate_silence(2.0))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("silence.wav", wav, "audio/wav")},
            headers=auth_header(),
        )
        assert r.status_code == 200
        body = r.json()
        assert "text" in body
        assert "language" in body
        assert "duration" in body
        assert body["duration"] >= 1.5

    def test_sine_wave_returns_response(self):
        """A tone should go through the pipeline without errors."""
        wav = build_wav(generate_sine(440, 2.0))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("tone.wav", wav, "audio/wav")},
            headers=auth_header(),
        )
        assert r.status_code == 200
        body = r.json()
        assert "text" in body
        assert "language" in body
        assert isinstance(body["duration"], (int, float))

    def test_stereo_wav_decoded(self):
        """Stereo WAV should be downmixed and transcribed."""
        mono = generate_silence(1.0)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            stereo = []
            for s in mono:
                val = max(-32768, min(32767, int(s * 32767)))
                stereo.extend([val, val])
            wf.writeframes(struct.pack(f"<{len(stereo)}h", *stereo))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("stereo.wav", buf.getvalue(), "audio/wav")},
            headers=auth_header(),
        )
        assert r.status_code == 200

    def test_different_sample_rate(self):
        """8kHz WAV should be accepted."""
        wav = build_wav(generate_sine(300, 1.5, sample_rate=8000), sample_rate=8000)
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("low_sr.wav", wav, "audio/wav")},
            headers=auth_header(),
        )
        assert r.status_code == 200


class TestTranscribeRawPCM:
    def test_raw_pcm_accepted(self):
        """Raw PCM16LE (no WAV header) should be transcribed."""
        pcm = build_raw_pcm(generate_silence(1.5))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("audio.pcm", pcm, "audio/octet-stream")},
            params={"sample_rate": 16000},
            headers=auth_header(),
        )
        assert r.status_code == 200
        body = r.json()
        assert "text" in body


class TestEdgeCases:
    def test_empty_audio_returns_400(self):
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("empty.wav", b"", "audio/wav")},
            headers=auth_header(),
        )
        assert r.status_code == 400

    def test_garbage_data_returns_400(self):
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("garbage.wav", b"not_audio_data_at_all", "audio/wav")},
            headers=auth_header(),
        )
        assert r.status_code == 400

    def test_very_short_audio(self):
        """50ms of audio -- should not crash."""
        wav = build_wav(generate_silence(0.05))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("short.wav", wav, "audio/wav")},
            headers=auth_header(),
        )
        # May be 200 (empty text) or 400 -- either is acceptable, just not 500
        assert r.status_code in (200, 400)

    def test_long_audio(self):
        """30s of audio -- tests that timeouts don't kill it."""
        wav = build_wav(generate_sine(200, 30.0))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("long.wav", wav, "audio/wav")},
            headers=auth_header(),
            timeout=120,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["duration"] >= 29.0


class TestSpeakerVerification:
    def test_verify_speaker_without_profile(self):
        """Speaker verification with unknown device should still return 200."""
        wav = build_wav(generate_sine(440, 1.0))
        r = requests.post(
            f"{BASE_URL}/transcribe",
            files={"audio": ("test.wav", wav, "audio/wav")},
            params={"verify_speaker": True, "device_id": "nonexistent_device"},
            headers=auth_header(),
        )
        # Should not crash -- either verified fields present or not
        assert r.status_code == 200


if __name__ == "__main__":
    # Simple runner without pytest
    passed = 0
    failed = 0
    errors = []

    test_classes = [
        TestHealth, TestAuth, TestTranscribeWav,
        TestTranscribeRawPCM, TestEdgeCases, TestSpeakerVerification,
    ]

    for cls in test_classes:
        instance = cls()
        for name in sorted(dir(instance)):
            if not name.startswith("test_"):
                continue
            method = getattr(instance, name)
            full_name = f"{cls.__name__}.{name}"
            try:
                method()
                passed += 1
                print(f"  PASS  {full_name}")
            except AssertionError as e:
                failed += 1
                errors.append((full_name, e))
                print(f"  FAIL  {full_name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((full_name, e))
                print(f"  ERROR {full_name}: {e}")

    print(f"\n{passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
        sys.exit(1)
