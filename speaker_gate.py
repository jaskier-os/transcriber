"""Speaker verification gate using SpeechBrain ECAPA-TDNN.

Per-device speaker profiles stored as {device_id}.npy in a profiles directory.
"""

import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)


class SpeakerGate:
    """Verifies that audio belongs to the enrolled speaker.

    Uses SpeechBrain's ECAPA-TDNN model (192-dim embeddings, trained on VoxCeleb).
    Language-independent -- works equally well with Russian and English speech.
    """

    def __init__(self, profiles_dir, threshold=0.45):
        self._profiles_dir = profiles_dir
        self._threshold = threshold
        self._model = None
        os.makedirs(profiles_dir, exist_ok=True)
        log.info("Speaker profiles directory: %s", profiles_dir)

    def load_model(self):
        """Lazy-load the ECAPA-TDNN encoder. Call once at startup."""
        if self._model is not None:
            return
        self._patch_compat()
        from speechbrain.inference.speaker import EncoderClassifier
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        log.info("SpeechBrain ECAPA-TDNN model loaded")

    @staticmethod
    def _patch_compat():
        """Patch compatibility issues between speechbrain 1.0.x and newer deps."""
        import torchaudio
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["ffmpeg"]

        import huggingface_hub
        import huggingface_hub.errors
        _orig_download = huggingface_hub.hf_hub_download
        import functools
        @functools.wraps(_orig_download)
        def _patched_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            try:
                return _orig_download(*args, **kwargs)
            except huggingface_hub.errors.EntryNotFoundError as exc:
                raise ValueError(str(exc)) from exc
        huggingface_hub.hf_hub_download = _patched_download

    def _profile_path(self, device_id):
        """Resolve profile path: {device_id}.npy, fallback to default.npy."""
        if device_id:
            path = os.path.join(self._profiles_dir, f"{device_id}.npy")
            if os.path.exists(path):
                return path
        default = os.path.join(self._profiles_dir, "default.npy")
        if os.path.exists(default):
            return default
        return None

    def _embed(self, audio):
        """Compute 192-dim embedding from float32 16kHz audio."""
        self.load_model()
        waveform = torch.from_numpy(audio).unsqueeze(0)
        embedding = self._model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()

    def enroll(self, device_id, audio_samples):
        """Enroll from a list of audio arrays. Averages embeddings, saves to disk.

        Args:
            device_id: str identifier for the device/user
            audio_samples: list of np.ndarray float32 at 16kHz

        Returns:
            dict with enrollment info
        """
        self.load_model()
        embeddings = []
        for i, audio in enumerate(audio_samples):
            emb = self._embed(audio)
            embeddings.append(emb)
            log.info("Enrollment sample %d/%d embedded (norm=%.3f)",
                     i + 1, len(audio_samples), np.linalg.norm(emb))

        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)

        filename = f"{device_id}.npy" if device_id else "default.npy"
        save_path = os.path.join(self._profiles_dir, filename)
        np.save(save_path, avg)
        log.info("Speaker profile saved to %s (dim=%d)", save_path, avg.shape[0])
        return {"embedding_dim": int(avg.shape[0]), "samples_used": len(audio_samples)}

    def verify(self, device_id, audio):
        """Check if audio matches the enrolled speaker.

        Args:
            device_id: str device identifier to look up profile
            audio: np.ndarray float32 at 16kHz

        Returns:
            (verified: bool, similarity: float)
        """
        profile_path = self._profile_path(device_id)
        if profile_path is None:
            log.warning("No speaker profile for device '%s', rejecting", device_id)
            return False, 0.0

        profile_embedding = np.load(profile_path)
        emb = self._embed(audio)
        emb = emb / np.linalg.norm(emb)
        similarity = float(np.dot(profile_embedding, emb))
        verified = similarity >= self._threshold
        log.info("Speaker verify [%s]: similarity=%.3f threshold=%.3f verified=%s",
                 device_id or "default", similarity, self._threshold, verified)
        return verified, similarity
