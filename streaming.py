"""Real-time streaming transcription using local agreement policy.

Based on ufal/whisper_streaming (https://github.com/ufal/whisper_streaming).
Instead of splitting audio on silence and transcribing chunks independently,
keeps a growing audio buffer and re-transcribes it periodically. Word-level
timestamps + local agreement (2 consecutive transcriptions must agree) produce
stable output that naturally filters hallucinations.

Silero VAD (ONNX) detects speech boundaries. Translation via NLLB HTTP API.
"""
import asyncio
import json
import logging
import re
import threading
import time
from collections import deque

import aiohttp
import numpy as np

log = logging.getLogger("transcriber.streaming")

# Common Whisper hallucinations on short/noisy audio
_HALLUCINATIONS = frozenset({
    "thank you", "thank you.", "thanks.", "thanks for watching.",
    "thank you for watching.", "bye.", "bye", "bye-bye.", "bye-bye",
    "you", "the end.", "thanks for watching!", "thank you for watching!",
    "subtitles by the amara.org community",
})


def _is_hallucination(text: str) -> bool:
    cleaned = text.lower().strip()
    if cleaned in _HALLUCINATIONS or cleaned.rstrip(".!,") in _HALLUCINATIONS:
        return True
    fragments = [f.strip().rstrip(".!,") for f in cleaned.replace("!", ".").replace("?", ".").split(".") if f.strip()]
    if fragments and all(f in _HALLUCINATIONS or f == "" for f in fragments):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation (.!?) boundaries."""
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


class HypothesisBuffer:
    """Local agreement policy for stable streaming transcription.

    Adapted from ufal/whisper_streaming. Compares consecutive transcriptions
    word-by-word and only commits text that appears in both (longest common
    prefix of 2 last inserts).
    """

    def __init__(self):
        self.commited_in_buffer = deque()
        self.buffer = deque()
        self.new = deque()
        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new, offset):
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = deque((a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1)
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for _ in range(i):
                                self.new.popleft()
                            break

    def flush(self):
        """Returns longest common prefix of last 2 inserts as committed."""
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if not self.buffer:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.popleft()
                self.new.popleft()
            else:
                break
        self.buffer = self.new
        self.new = deque()
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, t):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= t:
            self.commited_in_buffer.popleft()

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    """Streaming ASR using local agreement policy.

    Adapted from ufal/whisper_streaming. Re-transcribes a growing audio buffer
    with word timestamps. Local agreement (2 consecutive runs must match)
    determines which words are confirmed. Buffer is trimmed at segment
    boundaries to prevent unbounded growth.
    """

    SAMPLING_RATE = 16000
    def __init__(self, whisper_model, language=None, buffer_trimming_sec=15, beam_size=1):
        self.whisper_model = whisper_model
        self.language = language
        self.buffer_trimming_sec = buffer_trimming_sec
        self.beam_size = beam_size
        self.init()

    def init(self, offset=None):
        self._audio_chunks = []
        self._audio_len = 0
        self._audio_buffer_dirty = True
        self._audio_buffer_cache = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer()
        self.buffer_time_offset = 0 if offset is None else offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.committed = []

    @property
    def audio_buffer(self):
        if self._audio_buffer_dirty:
            if self._audio_chunks:
                self._audio_buffer_cache = np.concatenate(self._audio_chunks)
            else:
                self._audio_buffer_cache = np.array([], dtype=np.float32)
            self._audio_buffer_dirty = False
        return self._audio_buffer_cache

    @audio_buffer.setter
    def audio_buffer(self, value):
        self._audio_chunks = [value]
        self._audio_len = len(value)
        self._audio_buffer_dirty = False
        self._audio_buffer_cache = value

    def insert_audio_chunk(self, audio):
        self._audio_chunks.append(audio)
        self._audio_len += len(audio)
        self._audio_buffer_dirty = True

    def _prompt(self):
        """Build initial_prompt from committed text (up to 200 chars)."""
        k = max(0, len(self.committed) - 1)
        while k > 0 and self.committed[k - 1][1] > self.buffer_time_offset:
            k -= 1
        p = [t for _, _, t in self.committed[:k]]
        prompt = []
        length = 0
        while p and length < 200:
            x = p.pop(-1)
            length += len(x) + 1
            prompt.append(x)
        return "".join(prompt[::-1])

    def process_iter(self):
        """Transcribe buffer, apply agreement. Returns (new_committed_text, incomplete_text)."""
        prompt = self._prompt()
        buf_seconds = self._audio_len / self.SAMPLING_RATE
        if buf_seconds < 0.1:
            return "", self._incomplete_text()

        log.debug("Transcribing %.2fs from offset %.2f", buf_seconds, self.buffer_time_offset)

        t0 = time.monotonic()
        try:
            segments, info = self.whisper_model.transcribe(
                self.audio_buffer,
                language=self.language or None,
                initial_prompt=prompt or None,
                beam_size=self.beam_size,
                word_timestamps=True,
                condition_on_previous_text=True,
                vad_filter=False,
            )
            segments = list(segments)
        except Exception as e:
            log.error("Transcription error: %s", e)
            return "", self._incomplete_text()
        infer_time = time.monotonic() - t0
        log.debug("Transcription took %.2fs for %.2fs of audio (ratio: %.1fx)",
                  infer_time, buf_seconds, infer_time / max(buf_seconds, 0.01))

        # Extract timestamped words, skip high no-speech-prob segments
        tsw = []
        for seg in segments:
            if seg.no_speech_prob > 0.9:
                continue
            for w in seg.words:
                tsw.append((w.start, w.end, w.word))

        # Local agreement
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        new_committed = self.transcript_buffer.flush()
        self.committed.extend(new_committed)

        # Buffer trimming at segment boundaries
        if buf_seconds > self.buffer_trimming_sec:
            self._chunk_completed_segment(segments)

        committed_text = "".join(t for _, _, t in new_committed)
        return committed_text, self._incomplete_text()

    def finish(self):
        """Flush remaining incomplete text (call when speech ends)."""
        incomplete = self.transcript_buffer.complete()
        text = "".join(t for _, _, t in incomplete)
        self.buffer_time_offset += self._audio_len / self.SAMPLING_RATE
        return text

    def _incomplete_text(self):
        return "".join(t for _, _, t in self.transcript_buffer.complete())

    def _chunk_completed_segment(self, segments):
        if not self.committed:
            return
        ends = [s.end for s in segments]
        t = self.committed[-1][1]
        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                log.debug("Buffer trimmed at segment boundary %.2f (was %.2fs)",
                          e, self._audio_len / self.SAMPLING_RATE)
                self._chunk_at(e)
                return
        trim_at = self.committed[-1][1]
        log.debug("Buffer trimmed at last committed word %.2f (was %.2fs)",
                  trim_at, self._audio_len / self.SAMPLING_RATE)
        self._chunk_at(trim_at)

    def _chunk_at(self, t):
        self.transcript_buffer.pop_commited(t)
        cut = t - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut * self.SAMPLING_RATE):]
        self.buffer_time_offset = t


# ---------------------------------------------------------------------------
# Translation mode: sentence-level segmentation + per-sentence translation
# ---------------------------------------------------------------------------

class SentenceSegmenter:
    """Splits committed ASR text into sentence-level segments for translation.

    A new segment starts when:
    - A sentence boundary (.!?) is detected in committed text
    - VAD silence >= sentence_silence_threshold (force_finalize)
    """

    def __init__(self):
        self.seg_id = 1
        self._current = ""  # growing text for current (partial) segment

    def feed(self, committed: str, incomplete: str) -> list[dict]:
        """Feed new committed + incomplete text. Returns segment updates.

        Returns list of {segId, text, partial} dicts. Complete sentences
        get their own finalized segment; the trailing fragment is partial.
        """
        self._current += committed
        updates = []

        # Split committed text on sentence boundaries
        sentences = _split_sentences(self._current)
        if len(sentences) > 1:
            # All but the last are complete sentences -> finalize them
            for sent in sentences[:-1]:
                updates.append({"segId": self.seg_id, "text": sent.strip(), "partial": False})
                self.seg_id += 1
            self._current = sentences[-1]

        # Current committed fragment only (no incomplete -- avoids text moving between segments)
        partial_text = self._current.strip()
        if partial_text:
            updates.append({"segId": self.seg_id, "text": partial_text, "partial": True})

        return updates

    def force_finalize(self) -> list[dict]:
        """Finalize current segment on silence gap. Returns updates."""
        text = self._current.strip()
        if not text:
            return []
        updates = [{"segId": self.seg_id, "text": text, "partial": False}]
        self.seg_id += 1
        self._current = ""
        return updates

    def finalize_all(self, remaining: str) -> list[dict]:
        """Finalize everything (session end). Returns updates."""
        self._current += remaining
        text = self._current.strip()
        if not text:
            return []
        # Split into sentences, finalize each
        sentences = _split_sentences(text)
        updates = []
        for sent in sentences:
            updates.append({"segId": self.seg_id, "text": sent.strip(), "partial": False})
            self.seg_id += 1
        self._current = ""
        return updates


class SegmentTranslator:
    """Per-sentence translation with change detection and recency window.

    Rules:
    - Only translate if seg_id is within the 2 most recent segments
    - Only translate if source text changed since last translation
    """

    def __init__(self, http_session, translator_url, api_key, source_nllb, target_nllb):
        self._http = http_session
        self._url = translator_url
        self._api_key = api_key
        self._source_nllb = source_nllb
        self._target_nllb = target_nllb
        # {seg_id: {"source": str, "translation": str, "last_source": str}}
        self._segments = {}
        self._max_seg_id = 0

    async def translate_segment(self, seg_id: int, text: str) -> str:
        """Translate a segment if rules allow. Returns translation or ""."""
        self._max_seg_id = max(self._max_seg_id, seg_id)

        # Rule: only translate segments within 2 most recent
        if seg_id < self._max_seg_id - 1:
            seg = self._segments.get(seg_id)
            return seg["translation"] if seg else ""

        if seg_id not in self._segments:
            self._segments[seg_id] = {"source": "", "translation": "", "last_source": ""}

        seg = self._segments[seg_id]

        # Rule: only translate if text changed
        if text == seg["last_source"]:
            return seg["translation"]

        # Translate
        translation = await self._do_translate(text)
        if translation:
            seg["source"] = text
            seg["translation"] = translation
            seg["last_source"] = text
            return translation

        # Translation failed -- keep previous
        return seg["translation"]

    async def _do_translate(self, text: str) -> str | None:
        if not self._http or self._http.closed or not text.strip():
            return None
        try:
            headers = {}
            if self._api_key:
                headers["x-api-key"] = self._api_key
            url = f"{self._url}/translate"
            payload = {"text": text, "source_lang": self._source_nllb, "target_lang": self._target_nllb}
            log.info("Translate POST %s key=%s nllb=%s->%s text=%s", url, self._api_key[:8] if self._api_key else "NONE", self._source_nllb, self._target_nllb, text[:30])
            async with self._http.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=4.5),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.error("Translation HTTP %d: %s", resp.status, body[:200])
                    return None
                data = await resp.json()
                translations = data.get("translations", [])
                if translations:
                    return translations[0].get("text", "").strip() or None
        except Exception as e:
            log.error("Translation error (%s): %s", type(e).__name__, e)
        return None

    def get_cached(self, seg_id: int) -> str:
        """Get cached translation for a segment (no HTTP call)."""
        seg = self._segments.get(seg_id)
        return seg["translation"] if seg else ""


class StreamingSession:
    """Real-time streaming transcription session with translation.

    Uses OnlineASRProcessor (local agreement policy) for stable transcription.
    Silero VAD (ONNX) detects speech start/end. All audio during speech is fed
    to the processor; on silence the segment is finalized. Translation via NLLB.

    Translation mode: sentence-level segmentation with per-sentence translation.
    Each output message bundles source text + translation for a single sentence.
    """

    def __init__(self, ws, whisper_model, config: dict, vad_session=None):
        self.ws = ws
        self.whisper_model = whisper_model
        self.translator_url = config.get("translator_url", "http://localhost:10015")
        self.api_key = config.get("api_key", "")
        self.source_lang = config.get("source_lang", "en")
        self.target_lang = config.get("target_lang", "ru")
        self.source_nllb = config.get("source_nllb") or "eng_Latn"
        self.target_nllb = config.get("target_nllb") or "rus_Cyrl"
        self.translate = config.get("translate", False)
        self.sample_rate = config.get("sample_rate", 16000)
        self.min_chunk_size = config.get("min_chunk_size", 1.0)
        self.silence_threshold = config.get("silence_threshold", 1.0)
        self.sentence_silence = config.get("sentence_silence", 1.5)

        self.seg_id = 1

        # ASR processor with local agreement
        self._processor = OnlineASRProcessor(
            whisper_model,
            language=self.source_lang,
            buffer_trimming_sec=8,
            beam_size=config.get("beam_size", 1),
        )

        # Silero VAD for speech boundary detection
        self._vad_session = vad_session
        self._vad_threshold = config.get("vad_threshold", 0.5)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)
        self._vad_buffer = np.array([], dtype=np.float32)
        self._vad_sr = np.array(self.sample_rate, dtype=np.int64)
        self._vad_neg_threshold = max(self._vad_threshold - 0.15, 0.01)

        # Voice activity state
        self._finalizing = False
        self._status = "nonvoice"
        self._consecutive_silence_windows = 0
        self._last_audio_time = 0.0
        self._pre_speech_buffer = deque(maxlen=8)
        self._current_chunk_samples = 0

        # GPU serialization
        self._gpu_lock = threading.Lock()

        self._running = True
        self._http_session = None
        self._bg_task = None

        # Non-translation mode: simple segment tracking
        self._committed_text = ""
        self._last_sent_text = ""

        # Translation mode: sentence segmentation + per-sentence translation
        self._segmenter = None
        self._translator = None
        self._partial_count = 0  # translate every 2nd partial
        self._audio_dump = None
        self._audio_dump_path = config.get("audio_dump_path", "")

    async def start(self):
        # Optional TLS to translator: custom CA only if TLS_CERT_PATH is set,
        # otherwise system CA store. Plain http works with no cert at all.
        from config import build_ssl_context
        ssl_ctx = build_ssl_context()
        conn = aiohttp.TCPConnector(ssl=ssl_ctx)
        self._http_session = aiohttp.ClientSession(connector=conn)
        self._last_audio_time = time.monotonic()
        if self._audio_dump_path:
            try:
                self._audio_dump = open(self._audio_dump_path, "wb")
                log.info("Audio dump: writing raw PCM to %s", self._audio_dump_path)
            except Exception as e:
                log.error("Audio dump failed to open: %s", e)
        if self.translate:
            self._segmenter = SentenceSegmenter()
            self._translator = SegmentTranslator(
                self._http_session, self.translator_url, self.api_key,
                self.source_nllb, self.target_nllb,
            )
        log.info("Streaming session started (source=%s, target=%s, translate=%s, chunk=%.1fs, silence=%.1fs, sent_silence=%.1fs)",
                 self.source_lang, self.target_lang, self.translate,
                 self.min_chunk_size, self.silence_threshold, self.sentence_silence)
        self._bg_task = asyncio.create_task(self._bg_loop())

    async def stop(self):
        self._running = False
        if self._bg_task:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self._bg_task = None
        # Finalize remaining speech
        if self._status == "voice":
            text = self._finish_sync()
            if self.translate:
                remaining = (self._committed_text + text).strip()
                if remaining and not _is_hallucination(remaining):
                    updates = self._segmenter.finalize_all(text)
                    for upd in updates:
                        translation = await self._translator.translate_segment(upd["segId"], upd["text"])
                        await self._send_segment(upd["segId"], upd["text"], translation, upd["partial"])
            else:
                full = (self._committed_text + text).strip()
                if full:
                    log.info("Stop finalize seg=%d: %s", self.seg_id, full[:200])
                    await self._send_final(full)
        if self._audio_dump:
            self._audio_dump.close()
            log.info("Audio dump saved: %s", self._audio_dump_path)
            self._audio_dump = None
        if self._http_session:
            await self._http_session.close()
        log.info("Streaming session stopped")

    def feed_audio(self, pcm_bytes: bytes):
        """Feed raw PCM16LE bytes. Non-blocking."""
        if not self._running:
            return
        # Debug: dump raw PCM to file for mic testing
        if self._audio_dump:
            try:
                self._audio_dump.write(pcm_bytes)
            except Exception:
                pass
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._last_audio_time = time.monotonic()
        has_speech = self._vad_detect_speech(samples)

        if self._status == "voice":
            self._processor.insert_audio_chunk(samples)
            self._current_chunk_samples += len(samples)
        elif not self._finalizing:
            self._pre_speech_buffer.append(samples)
            if has_speech:
                self._status = "voice"
                self._consecutive_silence_windows = 0
                self._processor.init()
                self._committed_text = ""
                self._last_sent_text = ""
                self._current_chunk_samples = 0
                for chunk in self._pre_speech_buffer:
                    self._processor.insert_audio_chunk(chunk)
                    self._current_chunk_samples += len(chunk)
                self._pre_speech_buffer.clear()
                log.info("Speech started, seg=%d", self.seg_id)

    def _vad_detect_speech(self, samples: np.ndarray) -> bool:
        if self._vad_session is None:
            rms = float(np.sqrt(np.mean(samples ** 2)))
            return rms >= 0.015
        self._vad_buffer = np.concatenate([self._vad_buffer, samples])
        any_speech = False
        chunk_size = 512
        while len(self._vad_buffer) >= chunk_size:
            chunk = self._vad_buffer[:chunk_size].reshape(1, -1)
            self._vad_buffer = self._vad_buffer[chunk_size:]
            x = np.concatenate([self._vad_context, chunk], axis=1)
            ort_inputs = {"input": x, "state": self._vad_state, "sr": self._vad_sr}
            out, self._vad_state = self._vad_session.run(None, ort_inputs)
            self._vad_context = x[:, -64:]
            prob = out[0][0]
            if prob > self._vad_threshold:
                any_speech = True
                self._consecutive_silence_windows = 0
            elif prob < self._vad_neg_threshold:
                self._consecutive_silence_windows += 1
        return any_speech

    # --- background processing loop ---

    async def _bg_loop(self):
        try:
            while self._running:
                await asyncio.sleep(0.1)

                if self._status != "voice":
                    continue

                frame_silence = self._consecutive_silence_windows * 512 / self.sample_rate
                wall_silence = (time.monotonic() - self._last_audio_time) if self._last_audio_time else 0
                silence_dur = max(frame_silence, wall_silence)

                if self.translate:
                    await self._bg_loop_translate(silence_dur)
                else:
                    await self._bg_loop_plain(silence_dur)
        except asyncio.CancelledError:
            pass

    async def _bg_loop_plain(self, silence_dur: float):
        """Non-translation mode: original behavior."""
        if silence_dur >= self.silence_threshold:
            self._status = "nonvoice"
            self._finalizing = True
            self._current_chunk_samples = 0
            self._consecutive_silence_windows = 0
            log.info("Speech ended (silence=%.1fs), finalizing seg=%d",
                     silence_dur, self.seg_id)
            text = await asyncio.get_running_loop().run_in_executor(
                None, self._finish_sync)
            self._finalizing = False
            full = (self._committed_text + text).strip()
            if full:
                if _is_hallucination(full):
                    log.info("Filtered hallucination seg=%d: %s", self.seg_id, full)
                else:
                    await self._send_final(full)
            self.seg_id += 1

        elif self._current_chunk_samples >= self.sample_rate * self.min_chunk_size:
            committed, incomplete = await asyncio.get_running_loop().run_in_executor(
                None, self._process_iter_sync)
            if committed:
                self._committed_text += committed
                log.info("Committed seg=%d: +[%s] (total: [%s])",
                         self.seg_id, committed.strip()[:80],
                         self._committed_text.strip()[:120])
            full = (self._committed_text + incomplete).strip()
            if full and full != self._last_sent_text:
                if not _is_hallucination(full):
                    self._last_sent_text = full
                    await self._send_json({
                        "type": "transcription",
                        "segId": self.seg_id,
                        "text": full,
                        "partial": True,
                    })
            self._current_chunk_samples = 0

    async def _bg_loop_translate(self, silence_dur: float):
        """Translation mode: sentence-level segmentation + per-sentence translation."""

        # Sentence-level silence finalization (1.5s)
        if silence_dur >= self.sentence_silence:
            if silence_dur >= self.silence_threshold:
                # Full speech end: finalize everything
                self._status = "nonvoice"
                self._finalizing = True
                self._current_chunk_samples = 0
                self._consecutive_silence_windows = 0
                log.info("Speech ended (silence=%.1fs), finalizing", silence_dur)
                text = await asyncio.get_running_loop().run_in_executor(
                    None, self._finish_sync)
                self._finalizing = False
                remaining = text.strip()
                if remaining and not _is_hallucination(self._committed_text + remaining):
                    updates = self._segmenter.finalize_all(remaining)
                    for upd in updates:
                        translation = await self._translator.translate_segment(upd["segId"], upd["text"])
                        await self._send_segment(upd["segId"], upd["text"], translation, False)
                return
            else:
                # Sentence-level silence: flush incomplete text into segmenter, then finalize
                text = await asyncio.get_running_loop().run_in_executor(
                    None, self._finish_sync)
                if text:
                    self._committed_text += text
                    # Feed the flushed text as committed (it's confirmed by silence)
                    self._segmenter._current += text
                # Re-init processor to continue listening (finish() resets it)
                self._processor.init(offset=self._processor.buffer_time_offset)
                updates = self._segmenter.force_finalize()
                for upd in updates:
                    translation = await self._translator.translate_segment(upd["segId"], upd["text"])
                    await self._send_segment(upd["segId"], upd["text"], translation, False)
                    log.info("Sentence finalized (silence=%.1fs) seg=%d: %s",
                             silence_dur, upd["segId"], upd["text"][:80])
                self._current_chunk_samples = 0

        if self._current_chunk_samples >= self.sample_rate * self.min_chunk_size:
            committed, incomplete = await asyncio.get_running_loop().run_in_executor(
                None, self._process_iter_sync)
            if committed:
                self._committed_text += committed
                log.info("Committed: +[%s] (total: [%s])",
                         committed.strip()[:80],
                         self._committed_text.strip()[:120])

            # Feed to sentence segmenter
            updates = self._segmenter.feed(committed, incomplete)
            for upd in updates:
                if upd["partial"]:
                    self._partial_count += 1
                    if self._partial_count % 2 == 0:
                        # Every 2nd partial: translate then send
                        translation = await self._translator.translate_segment(upd["segId"], upd["text"])
                        await self._send_segment(upd["segId"], upd["text"], translation, True)
                    else:
                        # Odd partials: send with cached translation (no HTTP call)
                        cached = self._translator.get_cached(upd["segId"])
                        await self._send_segment(upd["segId"], upd["text"], cached, True)
                else:
                    # Complete sentence: translate and send
                    if _is_hallucination(upd["text"]):
                        log.info("Filtered hallucination seg=%d: %s", upd["segId"], upd["text"])
                        continue
                    translation = await self._translator.translate_segment(upd["segId"], upd["text"])
                    await self._send_segment(upd["segId"], upd["text"], translation, False)
                    log.info("Sentence complete seg=%d: %s -> %s",
                             upd["segId"], upd["text"][:60], translation[:60] if translation else "")

            self._current_chunk_samples = 0

    def _process_iter_sync(self):
        with self._gpu_lock:
            return self._processor.process_iter()

    def _finish_sync(self):
        with self._gpu_lock:
            return self._processor.finish()

    # --- WebSocket output ---

    async def _send_segment(self, seg_id: int, text: str, translation: str, partial: bool):
        """Send a bundled source+translation segment update."""
        msg = {
            "type": "transcription" if partial else "final",
            "segId": seg_id,
            "text": text,
            "partial": partial,
        }
        if translation:
            msg["translation"] = translation
        log.info("%s seg=%d: %s | %s",
                 "Partial" if partial else "Final", seg_id,
                 text[:80], translation[:40] if translation else "")
        await self._send_json(msg)

    async def _send_final(self, text: str):
        """Send final for non-translation mode."""
        log.info("Final seg=%d: %s", self.seg_id, text[:200])
        await self._send_json({
            "type": "final",
            "segId": self.seg_id,
            "text": text,
            "partial": False,
        })

    async def _send_json(self, data: dict):
        try:
            await self.ws.send_text(json.dumps(data))
        except (ConnectionError, OSError, RuntimeError) as e:
            log.error("WS send error: %s", e)
            self._running = False
