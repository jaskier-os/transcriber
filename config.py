import os
import logging

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
_device_raw = os.environ.get("DEVICE", "cpu").lower()
DEVICE = "cuda" if _device_raw in ("gpu", "cuda") else "cpu"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
PORT = int(os.environ.get("PORT", "10003"))
BEAM_SIZE = int(os.environ.get("BEAM_SIZE", "1"))
LANGUAGE = os.environ.get("LANGUAGE", "ru") or None
CPU_THREADS = int(os.environ.get("CPU_THREADS", "0")) or 0

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required")

SPEAKER_ENABLED = os.environ.get("SPEAKER_ENABLED", "false").lower() in ("true", "1", "yes")
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.45"))
SPEAKER_PROFILES_DIR = os.environ.get("SPEAKER_PROFILES_DIR", "./profiles")

ORCHESTRATOR_WS_URL = os.environ.get("ORCHESTRATOR_WS_URL", "")  # e.g. wss://orchestrator.example.com:8443/ws/transcriber

TRANSLATOR_URL = os.environ.get("TRANSLATOR_URL", "http://localhost:10015")

# Optional path to a custom CA bundle (PEM) for verifying TLS connections to the
# orchestrator / translator. Leave unset to use the system CA store and default
# verification. The service runs fine with no cert and over plain connections.
TLS_CERT_PATH = os.environ.get("TLS_CERT_PATH", "")
STREAMING_VAD_THRESHOLD = float(os.environ.get("STREAMING_VAD_THRESHOLD", "0.5"))
STREAMING_MIN_CHUNK_SIZE = float(os.environ.get("STREAMING_MIN_CHUNK_SIZE", "1.0"))
STREAMING_SILENCE_THRESHOLD = float(os.environ.get("STREAMING_SILENCE_THRESHOLD", "3.0"))
STREAMING_SENTENCE_SILENCE = float(os.environ.get("STREAMING_SENTENCE_SILENCE", "1.5"))


def build_ssl_context():
    """Build an SSL context for outbound TLS (wss/https) connections.

    Certificates and VPN are optional. If TLS_CERT_PATH points to a readable PEM
    bundle it is loaded as an additional trust anchor; otherwise the system CA
    store and default verification are used. Plain (non-TLS) connections need no
    context and never call this.
    """
    import ssl

    ctx = ssl.create_default_context()
    if TLS_CERT_PATH and os.path.exists(TLS_CERT_PATH):
        try:
            ctx.load_verify_locations(cafile=TLS_CERT_PATH)
            log.info("Loaded custom CA bundle from %s", TLS_CERT_PATH)
        except Exception as exc:
            log.warning("Failed to load TLS_CERT_PATH '%s': %s; using system CAs", TLS_CERT_PATH, exc)
    return ctx
