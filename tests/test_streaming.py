"""
Simulate the FULL phone->orchestrator->transcriber pipeline for glasses translation.

Replicates exactly what the phone does:
1. Connects to orchestrator /ws/transcribe (relay proxy)
2. Sends start + speech_start JSON messages
3. Glasses mic audio arrives as 512-sample (1024-byte) chunks via BT
4. Phone applies 4x gain (glasses already applied 16x = total 64x)
5. Phone batches into 3200-byte (100ms) frames before sending via WS
6. Results flow back: transcriber -> orchestrator -> phone -> glasses BT

Logs every step the user would see on the glasses.

Usage:
  python test_streaming.py [--direct]    # via orchestrator relay (default)
  python test_streaming.py --direct      # direct to transcriber (bypass orchestrator)
"""
import asyncio
import json
import ssl
import struct
import sys
import time
import os

import numpy as np
import websockets

PCM_FILE = os.path.join(os.path.dirname(__file__), "test_30s.pcm")
PCM_FILE_PART2 = os.path.join(os.path.dirname(__file__), "test_30s_part2.pcm")
PCM_FILE_PART3 = os.path.join(os.path.dirname(__file__), "test_120s.pcm")
TIMESTAMPS_FILE = os.path.join(os.path.dirname(__file__), "audio_timestamps.json")
# Default: connect via orchestrator relay (same as phone)
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "ws://localhost:10001/ws/transcribe")
DIRECT_URL = os.environ.get("DIRECT_URL", "ws://localhost:10003/ws/stream")
SAMPLE_RATE = 16000
# Phone batches audio into 3200-byte frames (100ms at 16kHz 16-bit mono)
BATCH_BYTES = 3200
# Glasses BT sends 512-sample chunks (1024 bytes of PCM16LE)
GLASSES_CHUNK_SAMPLES = 512
# Gain applied by phone (glasses already apply 16x, phone adds 4x = 64x total)
# NOTE: Set to 1 for normal test files. Only use 4 for actual Rokid mic recordings
# which arrive at ~1% full scale.
PHONE_GAIN = 1
# Pauses between parts (simulates user pausing mic then resuming)
PAUSE_AFTER_PART1 = 20
PAUSE_AFTER_PART2 = 10
API_KEY = os.environ.get("API_KEY", "")
# Translation settings
SOURCE_LANG = os.environ.get("SOURCE_LANG", "en")
TARGET_LANG = os.environ.get("TARGET_LANG", "ru")
SOURCE_NLLB = os.environ.get("SOURCE_NLLB", "eng_Latn")
TARGET_NLLB = os.environ.get("TARGET_NLLB", "rus_Cyrl")
TRANSLATE = "--no-translate" not in sys.argv


def apply_gain(pcm_bytes: bytes, gain: int) -> bytes:
    """Apply gain to PCM16LE audio, clamping to int16 range. Same as phone's feedGlassesAudioFromMic()."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.int32)
    samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
    return samples.tobytes()


async def main():
    direct = "--direct" in sys.argv
    url = DIRECT_URL if direct else ORCHESTRATOR_URL
    pcm_path = PCM_FILE

    # Allow positional args for pcm file
    for arg in sys.argv[1:]:
        if not arg.startswith("--") and os.path.exists(arg):
            pcm_path = arg

    with open(pcm_path, "rb") as f:
        pcm_data = f.read()

    # Load additional parts if they exist (for pause-resume test)
    pcm_data_part2 = None
    pcm_data_part3 = None
    has_part2 = os.path.exists(PCM_FILE_PART2) and pcm_path == PCM_FILE
    has_part3 = os.path.exists(PCM_FILE_PART3) and pcm_path == PCM_FILE
    if has_part2:
        with open(PCM_FILE_PART2, "rb") as f:
            pcm_data_part2 = f.read()
    if has_part3:
        with open(PCM_FILE_PART3, "rb") as f:
            pcm_data_part3 = f.read()

    total_samples = len(pcm_data) // 2
    total_duration = total_samples / SAMPLE_RATE
    if pcm_data_part2:
        total_duration += PAUSE_AFTER_PART1 + len(pcm_data_part2) // 2 / SAMPLE_RATE
    if pcm_data_part3:
        total_duration += PAUSE_AFTER_PART2 + len(pcm_data_part3) // 2 / SAMPLE_RATE

    print(f"{'='*80}")
    print(f"GLASSES TRANSLATION PIPELINE SIMULATION")
    print(f"{'='*80}")
    print(f"Audio     : {pcm_path} ({len(pcm_data)//2/SAMPLE_RATE:.1f}s, {len(pcm_data)} bytes)")
    if pcm_data_part2:
        print(f"Part 2    : {PCM_FILE_PART2} ({len(pcm_data_part2)//2/SAMPLE_RATE:.1f}s, pause={PAUSE_AFTER_PART1}s before)")
    if pcm_data_part3:
        print(f"Part 3    : {PCM_FILE_PART3} ({len(pcm_data_part3)//2/SAMPLE_RATE:.1f}s, pause={PAUSE_AFTER_PART2}s before)")
    print(f"Endpoint  : {url} ({'DIRECT to transcriber' if direct else 'via orchestrator relay'})")
    print(f"Gain      : {PHONE_GAIN}x (phone) -- glasses already applied 16x = {PHONE_GAIN*16}x total")
    print(f"Batch size: {BATCH_BYTES} bytes ({BATCH_BYTES / (SAMPLE_RATE * 2) * 1000:.0f}ms)")
    print(f"{'='*80}")
    print()

    # What the user sees on glasses (timeline of display events)
    glasses_display = []  # (time, event_type, seg_id, text)
    results = []
    t0 = time.monotonic()

    def elapsed():
        return time.monotonic() - t0

    def log(msg):
        print(f"  [{elapsed():6.2f}s] {msg}")

    try:
        connect_kwargs = {"open_timeout": 10, "max_size": None}
        if url.startswith("wss"):
            connect_kwargs["ssl"] = ssl.create_default_context()
        if API_KEY:
            connect_kwargs["additional_headers"] = {"x-api-key": API_KEY}

        async with websockets.connect(url, **connect_kwargs) as ws:
            log("WS connected")

            # Step 1: Send start message (same as OrchestratorClient.connectTranscribeStream)
            start_msg = json.dumps({
                "type": "start",
                "sampleRate": SAMPLE_RATE,
                "sourceLang": SOURCE_LANG,
                "targetLang": TARGET_LANG,
                "sourceNllb": SOURCE_NLLB,
                "targetNllb": TARGET_NLLB,
                "translate": TRANSLATE,
            })
            await ws.send(start_msg)
            log("PHONE: sent 'start' message")

            # Step 2: Send speech_start immediately (phone does this right after start)
            speech_start_msg = json.dumps({"type": "speech_start"})
            await ws.send(speech_start_msg)
            log("PHONE: sent 'speech_start' message")

            # Step 3: Background receiver -- simulates what glasses display shows
            async def receiver():
                try:
                    while True:
                        msg = await ws.recv()
                        t = elapsed()
                        data = json.loads(msg)
                        results.append((t, data))
                        msg_type = data.get("type", "?")
                        seg_id = data.get("segId", 0)

                        if msg_type == "transcription":
                            text = data.get("text", "")
                            partial = data.get("partial", True)
                            tag = "PARTIAL" if partial else "FINAL"
                            # This is what phone sends to glasses via sendTranslationResultToApp
                            glasses_display.append((t, "text", seg_id, text, partial))
                            log(f"GLASSES <- {tag:7s} seg={seg_id}: {text[:120]}")

                        elif msg_type == "final":
                            text = data.get("text", "")
                            glasses_display.append((t, "text", seg_id, text, False))
                            log(f"GLASSES <- FINAL   seg={seg_id}: {text[:120]}")

                        elif msg_type == "translation":
                            trans = data.get("translation", "")
                            glasses_display.append((t, "trans", seg_id, trans, True))
                            log(f"GLASSES <- TRANS   seg={seg_id}: {trans[:120]}")

                        elif msg_type == "error":
                            log(f"ERROR: {data.get('message', '')}")

                except websockets.ConnectionClosed:
                    pass

            recv_task = asyncio.create_task(receiver())

            # Step 4: Simulate glasses mic -> phone -> server audio pipeline
            # Glasses send 512-sample chunks over BT. Phone applies gain, batches to 3200 bytes.
            log("PHONE: starting audio stream (simulating glasses mic -> BT -> phone -> server)")

            audio_buf = bytearray()  # Phone's transcribeAudioBuf
            send_start = time.monotonic()
            offset = 0
            chunk_bytes = GLASSES_CHUNK_SAMPLES * 2  # 1024 bytes per glasses BT chunk
            frames_sent = 0

            while offset < len(pcm_data):
                # Simulate receiving a 512-sample chunk from glasses via BT
                chunk = pcm_data[offset:offset + chunk_bytes]
                offset += chunk_bytes

                # Phone applies gain (same as feedGlassesAudioFromMic)
                chunk = apply_gain(chunk, PHONE_GAIN)

                # Phone batches into BATCH_BYTES frames (same as sendTranscribeAudioFrame)
                audio_buf.extend(chunk)
                if len(audio_buf) >= BATCH_BYTES:
                    await ws.send(bytes(audio_buf[:BATCH_BYTES]))
                    audio_buf = audio_buf[BATCH_BYTES:]
                    frames_sent += 1

                # Pace to real-time (simulate real-time mic capture)
                elapsed_send = time.monotonic() - send_start
                expected_time = offset / (SAMPLE_RATE * 2)
                if expected_time > elapsed_send:
                    await asyncio.sleep(expected_time - elapsed_send)

            # Flush remaining audio (phone's flushTranscribeAudioBuf)
            if audio_buf:
                await ws.send(bytes(audio_buf))
                frames_sent += 1
                audio_buf.clear()

            log(f"PHONE: part 1 done ({frames_sent} frames, {offset} bytes raw)")

            # Helper to stream a PCM buffer in real-time
            async def stream_part(data, part_name):
                nonlocal frames_sent
                buf = bytearray()
                start = time.monotonic()
                off = 0
                while off < len(data):
                    chunk = data[off:off + chunk_bytes]
                    off += chunk_bytes
                    chunk = apply_gain(chunk, PHONE_GAIN)
                    buf.extend(chunk)
                    if len(buf) >= BATCH_BYTES:
                        await ws.send(bytes(buf[:BATCH_BYTES]))
                        buf = buf[BATCH_BYTES:]
                        frames_sent += 1
                    el = time.monotonic() - start
                    exp = off / (SAMPLE_RATE * 2)
                    if exp > el:
                        await asyncio.sleep(exp - el)
                if buf:
                    await ws.send(bytes(buf))
                    frames_sent += 1
                    buf.clear()
                log(f"PHONE: {part_name} done ({frames_sent} total frames)")

            # Part 2: pause then resume (tests stale buffer handling)
            if pcm_data_part2:
                log(f"PHONE: PAUSING for {PAUSE_AFTER_PART1}s (simulating user stops talking)")
                await asyncio.sleep(PAUSE_AFTER_PART1)
                log(f"PHONE: RESUMING with part 2 ({len(pcm_data_part2)//2/SAMPLE_RATE:.1f}s)")
                await stream_part(pcm_data_part2, "part 2")

            # Part 3: 2 minutes of uninterrupted audio (tests buffer trimming)
            if pcm_data_part3:
                log(f"PHONE: PAUSING for {PAUSE_AFTER_PART2}s")
                await asyncio.sleep(PAUSE_AFTER_PART2)
                log(f"PHONE: RESUMING with part 3 ({len(pcm_data_part3)//2/SAMPLE_RATE:.1f}s -- long continuous speech)")
                await stream_part(pcm_data_part3, "part 3")

            log(f"PHONE: all audio sent")

            # Wait for remaining results (longer for 120s part3)
            await asyncio.sleep(8)

            # Send stop (phone calls disconnectTranscribeStream)
            await ws.send(json.dumps({"type": "stop"}))
            log("PHONE: sent 'stop' message")
            await asyncio.sleep(2)
            recv_task.cancel()

    except Exception as e:
        print(f"\nCONNECTION ERROR: {e}")
        print(f"Make sure the {'orchestrator (port 10001)' if not direct else 'transcriber (port 10003)'} is running")
        return

    # ==================== SUMMARY ====================
    total_elapsed = elapsed()
    print()
    print(f"{'='*80}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total wall time : {total_elapsed:.1f}s (audio: {total_duration:.1f}s)")
    print(f"Overhead        : {total_elapsed - total_duration:.1f}s")
    print()

    partials = [(t, d) for t, d in results if d.get("type") == "transcription" and d.get("partial")]
    finals = [(t, d) for t, d in results if d.get("type") in ("final", "transcription") and not d.get("partial")]
    translations = [(t, d) for t, d in results if d.get("type") == "translation"]

    print(f"Partials: {len(partials)}, Finals: {len(finals)}, Translations: {len(translations)}")

    if partials:
        first_partial = partials[0]
        print(f"First partial    : {first_partial[0]:.2f}s (text: '{first_partial[1].get('text', '')[:60]}')")
    if finals:
        print(f"First final      : {finals[0][0]:.2f}s")
    if translations:
        print(f"First translation: {translations[0][0]:.2f}s")

    # Latency stats
    if len(partials) >= 2:
        gaps = [partials[i+1][0] - partials[i][0] for i in range(len(partials)-1)]
        avg_gap = sum(gaps) / len(gaps)
        print(f"Partial avg gap  : {avg_gap:.2f}s (min={min(gaps):.2f}s, max={max(gaps):.2f}s)")

    # Duplicate translation detection
    if translations:
        trans_texts = [d.get("translation", "") for _, d in translations]
        seen = set()
        dupes = []
        for i, t in enumerate(trans_texts):
            if t in seen:
                dupes.append((i, t))
            seen.add(t)
        print(f"\nTranslation duplicates: {len(dupes)}")
        for idx, t in dupes[:10]:
            print(f"  DUPE #{idx}: {t[:100]}")

        # Translation churn: how many times translation changed per segment
        seg_trans = {}
        for _, d in translations:
            sid = d.get("segId", 0)
            seg_trans.setdefault(sid, []).append(d.get("translation", ""))
        print(f"\nTranslation churn per segment:")
        for sid in sorted(seg_trans):
            changes = seg_trans[sid]
            print(f"  seg={sid}: {len(changes)} translation updates")
            for i, t in enumerate(changes):
                print(f"    [{i+1}] {t[:120]}")

    # What the user actually sees on glasses
    print()
    print(f"{'='*80}")
    print(f"GLASSES DISPLAY TIMELINE (what user sees)")
    print(f"{'='*80}")
    for t, evt_type, seg_id, text, partial in glasses_display:
        tag = "TEXT " if evt_type == "text" else "TRANS"
        state = "partial" if partial else "FINAL"
        print(f"  [{t:6.2f}s] [{tag}] seg={seg_id} ({state:7s}): {text[:140]}")

    # Final transcriptions
    print()
    print(f"--- Final transcriptions ---")
    full_text = []
    for t, data in finals:
        text = data.get("text", "")
        full_text.append(text)
        print(f"  [{t:5.1f}s] seg={data.get('segId')}: {text}")

    print(f"\n--- Full transcription ---")
    print(" ".join(full_text))

    print(f"\n--- Final translations ---")
    for t, data in translations:
        seg_id = data.get("segId")
        # Only show translation if it's the last one per segment
        print(f"  [{t:5.1f}s] seg={seg_id}: {data.get('translation', '')}")

    # ==================== WORD LATENCY ANALYSIS ====================
    if os.path.exists(TIMESTAMPS_FILE):
        with open(TIMESTAMPS_FILE) as f:
            ref = json.load(f)
        ref_words = ref.get("words", [])
        ref_sentences = ref.get("sentences", [])

        if ref_words and partials:
            print()
            print(f"{'='*80}")
            print(f"LATENCY ANALYSIS (vs reference word timestamps)")
            print(f"{'='*80}")
            print(f"  Audio streamed in real-time: word spoken at T in audio = sent at ~T wall-clock")
            print(f"  Latency = wall_clock_when_word_first_appears - word_audio_timestamp")
            print()

            # Sequential word matching: for each partial, match its words against
            # the reference word sequence in order. Track when each ref word is
            # first covered by a partial.
            def normalize(w):
                return w.lower().strip(".,!?'\":-")

            ref_normalized = [normalize(w["word"]) for w in ref_words]
            word_first_seen = {}  # ref_word_index -> wall_clock_time

            for pt, pd in partials + ([(finals[0][0], finals[0][1])] if finals else []):
                ptext = pd.get("text", "")
                pwords = [normalize(w) for w in ptext.split() if normalize(w)]
                # Match partial words against reference in sequence
                ri = 0
                for pw in pwords:
                    if ri < len(ref_normalized) and pw == ref_normalized[ri]:
                        if ri not in word_first_seen:
                            word_first_seen[ri] = pt
                        ri += 1
                    elif ri < len(ref_normalized):
                        # Skip non-matching words (filler, punctuation diffs)
                        # But try next ref word in case of inserted words
                        pass

            # Per-word latency table
            word_latencies = []
            print(f"  {'Word':<15s} {'Spoken':>7s} {'Received':>9s} {'Latency':>8s}")
            print(f"  {'-'*15} {'-'*7} {'-'*9} {'-'*8}")
            for i, rw in enumerate(ref_words):
                if i in word_first_seen:
                    latency = word_first_seen[i] - rw["start"]
                    word_latencies.append((rw["word"], rw["start"], word_first_seen[i], latency))
                    bar = "#" * max(0, min(40, int(latency)))
                    print(f"  {rw['word']:<15s} {rw['start']:6.1f}s  {word_first_seen[i]:7.1f}s  {latency:+6.1f}s  {bar}")
                else:
                    print(f"  {rw['word']:<15s} {rw['start']:6.1f}s  {'???':>7s}  {'???':>6s}")

            if word_latencies:
                lats = [l for _, _, _, l in word_latencies]
                print()
                print(f"  Latency stats: avg={sum(lats)/len(lats):.1f}s, "
                      f"min={min(lats):.1f}s, max={max(lats):.1f}s, "
                      f"median={sorted(lats)[len(lats)//2]:.1f}s")

            # Per-sentence latency: find when the LAST word of each sentence first appears
            print()
            print(f"  {'Sentence (first 60 chars)':<62s} {'Ends at':>7s} {'Received':>9s} {'Latency':>8s}")
            print(f"  {'-'*62} {'-'*7} {'-'*9} {'-'*8}")
            for sent in ref_sentences:
                # Find the last ref word that belongs to this sentence
                last_word_idx = None
                for i, rw in enumerate(ref_words):
                    if rw["start"] >= sent["start"] and rw["end"] <= sent["end"] + 0.1:
                        last_word_idx = i
                if last_word_idx is not None and last_word_idx in word_first_seen:
                    received = word_first_seen[last_word_idx]
                    latency = received - sent["end"]
                    print(f"  {sent['text'][:60]:<62s} {sent['end']:5.1f}s  {received:7.1f}s  {latency:+6.1f}s")
                else:
                    print(f"  {sent['text'][:60]:<62s} {sent['end']:5.1f}s  {'???':>7s}  {'???':>6s}")

            # Translation latency (per sentence - when does translation arrive after sentence ends)
            if translations:
                print()
                print(f"  {'Sentence (first 60 chars)':<62s} {'Ends at':>7s} {'1st trans':>10s} {'Latency':>8s}")
                print(f"  {'-'*62} {'-'*7} {'-'*10} {'-'*8}")
                for sent in ref_sentences:
                    found_time = None
                    for tt, td in translations:
                        if tt >= sent["end"]:
                            found_time = tt
                            break
                    if found_time is not None:
                        latency = found_time - sent["end"]
                        print(f"  {sent['text'][:60]:<62s} {sent['end']:5.1f}s  {found_time:8.1f}s  {latency:+6.1f}s")
                    else:
                        print(f"  {sent['text'][:60]:<62s} {sent['end']:5.1f}s  {'???':>8s}  {'???':>6s}")
    else:
        print(f"\n(No {TIMESTAMPS_FILE} found -- run generate_timestamps.py to enable latency analysis)")


if __name__ == "__main__":
    asyncio.run(main())
