"""
Outbound WebSocket client connecting to the orchestrator as a transcriber service.
The orchestrator relays phone audio through this connection.

Protocol (same as /ws/stream):
  Receive: start JSON, binary audio frames, stop JSON
  Send: partial/final/error JSON
"""
import asyncio
import json
import logging

import websockets

log = logging.getLogger("transcriber.bridge")

RECONNECT_DELAY = 5  # seconds between reconnect attempts


class OrchestratorBridge:
    def __init__(self, orchestrator_url: str, api_key: str, whisper_model, vad_session, streaming_config: dict):
        self.url = orchestrator_url
        self.api_key = api_key
        self.whisper_model = whisper_model
        self.vad_session = vad_session
        self.streaming_config = streaming_config
        self._task = None
        self._stop = False

    async def start(self):
        self._stop = False
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _build_ssl_context(self):
        """Build the outbound TLS context. Certificates/VPN are optional;
        falls back to the system CA store when no custom cert is configured."""
        from config import build_ssl_context
        return build_ssl_context()

    async def _run_loop(self):
        while not self._stop:
            try:
                await self._connect()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("Bridge connection error: %s", e)
            if not self._stop:
                log.info("Reconnecting in %ds...", RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)

    async def _connect(self):
        ssl_ctx = self._build_ssl_context()
        headers = {"x-api-key": self.api_key}
        log.info("Connecting to orchestrator: %s", self.url)

        async with websockets.connect(
            self.url,
            ssl=ssl_ctx if self.url.startswith("wss") else None,
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=10,
            max_size=None,
        ) as ws:
            log.info("Connected to orchestrator")
            await self._handle_connection(ws)

    async def _handle_connection(self, ws):
        from streaming import StreamingSession

        session = None

        class WsBridge:
            """Adapts websockets client to look like a FastAPI WebSocket for StreamingSession."""
            def __init__(self, ws_conn):
                self._ws = ws_conn

            async def send_text(self, text):
                try:
                    await self._ws.send(text)
                    log.info("Sent to orchestrator: %s", text[:80])
                except Exception as e:
                    log.error("Failed to send to orchestrator: %s", e)

            async def send_json(self, data):
                await self.send_text(json.dumps(data))

        bridge_ws = WsBridge(ws)

        frame_count = 0
        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    frame_count += 1
                    if frame_count == 1:
                        log.info("First audio frame: type=%s, len=%d", type(message).__name__, len(message))
                    if session:
                        session.feed_audio(bytes(message))
                        if frame_count % 100 == 1:
                            log.info("Fed frame #%d to session (%d bytes)", frame_count, len(message))
                    elif frame_count % 50 == 1:
                        log.warning("Audio frame #%d but no session", frame_count)
                else:
                    if frame_count == 0:
                        log.info("Control message (pre-audio): type=%s, len=%d, preview=%s", type(message).__name__, len(message), str(message)[:80])
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        log.warning("Failed to parse: %s", str(message)[:80])
                        continue

                    msg_type = data.get("type")

                    if msg_type == "start":
                        # Clean up previous session
                        if session:
                            try:
                                await session.stop()
                            except Exception:
                                pass

                        config = {
                            "sample_rate": data.get("sampleRate", 16000),
                            "source_lang": data.get("sourceLang") or None,
                            "target_lang": data.get("targetLang", "ru"),
                            "source_nllb": data.get("sourceNllb", "eng_Latn"),
                            "target_nllb": data.get("targetNllb", "rus_Cyrl"),
                            "translate": data.get("translate", False),
                            **self.streaming_config,
                        }

                        session = StreamingSession(
                            ws=bridge_ws,
                            whisper_model=self.whisper_model,
                            config=config,
                            vad_session=self.vad_session,
                        )
                        await session.start()
                        log.info("Transcription session started (lang=%s)", config.get("source_lang"))

                    elif msg_type == "stop":
                        if session:
                            try:
                                await session.stop()
                            except Exception:
                                pass
                            session = None
                            log.info("Transcription session stopped")

        except websockets.ConnectionClosed:
            log.info("Orchestrator connection closed")
        finally:
            if session:
                try:
                    await session.stop()
                except Exception:
                    pass
