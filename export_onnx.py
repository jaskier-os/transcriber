"""Export SpeechBrain ECAPA-TDNN encoder to ONNX for on-device speaker verification.

Exports only the embedding model, NOT the Fbank frontend or InputNormalization.
SpeechBrain's Fbank uses torch.stft (untraceable) and InputNormalization uses
running statistics that don't trace reliably.

Client-side preprocessing:
  1. Compute Fbank features (80 mel bins) from raw 16kHz audio
  2. Apply mean/variance normalization using exported stats
  3. Run ONNX encoder

ONNX model:
  Input:  "features" tensor [1, T, 80] (normalized Fbank features)
  Output: "embedding" tensor [1, 1, 192]

Produces:
  - ecapa_tdnn.onnx   (encoder: normalized fbank features -> 192-dim embedding)
  - voice_profile.bin  (raw float32 binary of the enrolled profile)
  - fbank_config.bin   (Fbank parameters + normalization stats for client-side preprocessing)

Usage:
    python export_onnx.py \
        --profile ../../clients/desktop/models/voice_profile.npy \
        --out-dir ../../clients/phone/app/src/main/assets
"""

import argparse
import logging
import os
import struct

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("export_onnx")


def patch_compat():
    """Same compat patches as speaker_gate.py."""
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]

    import functools
    import huggingface_hub
    import huggingface_hub.errors
    _orig = huggingface_hub.hf_hub_download

    @functools.wraps(_orig)
    def _patched(*args, **kwargs):
        kwargs.pop("use_auth_token", None)
        try:
            return _orig(*args, **kwargs)
        except huggingface_hub.errors.EntryNotFoundError as exc:
            raise ValueError(str(exc)) from exc

    huggingface_hub.hf_hub_download = _patched


class EcapaEncoderOnly(nn.Module):
    """Wraps just the embedding model (no Fbank, no MVN).

    Input:  normalized features tensor [1, T, 80]
    Output: embedding tensor [1, 1, 192]
    """

    def __init__(self, classifier):
        super().__init__()
        self.embedding_model = classifier.mods.embedding_model

    def forward(self, feats):
        return self.embedding_model(feats)


def load_classifier():
    patch_compat()
    from speechbrain.inference.speaker import EncoderClassifier
    log.info("Loading SpeechBrain ECAPA-TDNN...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    log.info("Model loaded")
    return classifier


def extract_fbank_config(classifier, out_path):
    """Extract Fbank and MVN parameters for client-side preprocessing.

    Binary format (all little-endian):
      - n_fft: int32
      - n_mels: int32
      - sample_rate: int32
      - f_min: float32
      - f_max: float32
      - win_length: int32
      - hop_length: int32
      - glob_mean: float32 (scalar global mean from InputNormalization)
      - glob_std: float32  (scalar global std from InputNormalization)
    """
    fbank = classifier.mods.compute_features

    # STFT config
    stft = fbank.compute_STFT
    n_fft = stft.n_fft
    sample_rate = stft.sample_rate
    win_length = stft.win_length
    hop_length = stft.hop_length

    # Mel filterbank config
    fbanks = fbank.compute_fbanks
    n_mels = fbanks.n_mels
    f_min = fbanks.f_min
    f_max = fbanks.f_max

    # MVN running stats (global scalars)
    mvn = classifier.mods.mean_var_norm
    glob_mean = float(mvn.glob_mean.item())
    glob_std = float(mvn.glob_std.item())

    log.info("Fbank: n_fft=%d n_mels=%d sr=%d f_min=%.1f f_max=%.1f win=%d hop=%d",
             n_fft, n_mels, sample_rate, f_min, f_max, win_length, hop_length)
    log.info("MVN: glob_mean=%.6f glob_std=%.6f", glob_mean, glob_std)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", n_fft))       # 400
        f.write(struct.pack("<i", n_mels))       # 80
        f.write(struct.pack("<i", sample_rate))  # 16000
        f.write(struct.pack("<f", f_min))        # 0.0
        f.write(struct.pack("<f", f_max))        # 8000.0
        f.write(struct.pack("<i", win_length))   # 400
        f.write(struct.pack("<i", hop_length))   # 160
        f.write(struct.pack("<f", glob_mean))
        f.write(struct.pack("<f", glob_std))

    log.info("Fbank config saved to %s (%d bytes)", out_path, os.path.getsize(out_path))


def export_onnx(classifier, out_path):
    wrapper = EcapaEncoderOnly(classifier)
    wrapper.eval()

    # Generate dummy normalized Fbank features: [1, T, 80]
    # 2s audio at 16kHz -> 25ms window, 10ms hop -> ~199 frames
    dummy_feats = torch.randn(1, 199, 80)

    log.info("Exporting encoder to ONNX: %s", out_path)

    torch.onnx.export(
        wrapper,
        (dummy_feats,),
        out_path,
        input_names=["features"],
        output_names=["embedding"],
        dynamic_axes={"features": {1: "num_frames"}},
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    log.info("ONNX export complete: %s (%.1f MB)", out_path, os.path.getsize(out_path) / 1e6)


def verify_onnx(classifier, onnx_path):
    """Verify ONNX encoder output matches PyTorch output exactly."""
    import onnxruntime as ort

    # Compute real Fbank features
    test_audio = torch.randn(1, 32000)
    with torch.no_grad():
        feats = classifier.mods.compute_features(test_audio)
        # Apply MVN exactly as SpeechBrain does
        norm_feats = classifier.mods.mean_var_norm(feats, torch.ones(1))

    # PyTorch embedding model
    with torch.no_grad():
        pt_emb = classifier.mods.embedding_model(norm_feats).squeeze().numpy()

    # ONNX embedding model
    sess = ort.InferenceSession(onnx_path)
    ort_emb = sess.run(["embedding"], {"features": norm_feats.numpy()})[0]
    ort_emb = ort_emb.squeeze()

    # Cosine similarity
    pt_norm = pt_emb / np.linalg.norm(pt_emb)
    ort_norm = ort_emb / np.linalg.norm(ort_emb)
    similarity = float(np.dot(pt_norm, ort_norm))
    log.info("PyTorch vs ONNX cosine similarity: %.6f", similarity)

    if similarity < 0.999:
        log.warning("Similarity below 0.999 -- outputs may diverge!")
    else:
        log.info("Verification passed")


def convert_profile(npy_path, bin_path):
    """Convert .npy voice profile to raw float32 binary."""
    profile = np.load(npy_path).astype(np.float32)
    log.info("Profile shape: %s, norm: %.4f", profile.shape, np.linalg.norm(profile))

    with open(bin_path, "wb") as f:
        f.write(profile.tobytes())

    log.info("Wrote %d bytes to %s (%d floats)",
             profile.shape[0] * 4, bin_path, profile.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Export ECAPA-TDNN encoder to ONNX")
    parser.add_argument(
        "--profile",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../../clients/desktop/models/voice_profile.npy"),
        help="Path to .npy voice profile",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../../clients/phone/app/src/main/assets"),
        help="Output directory for .onnx and .bin files",
    )
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip ONNX verification step")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "ecapa_tdnn.onnx")
    bin_path = os.path.join(args.out_dir, "voice_profile.bin")
    fbank_config_path = os.path.join(args.out_dir, "fbank_config.bin")

    classifier = load_classifier()
    export_onnx(classifier, onnx_path)
    extract_fbank_config(classifier, fbank_config_path)

    if not args.skip_verify:
        verify_onnx(classifier, onnx_path)

    if os.path.exists(args.profile):
        convert_profile(args.profile, bin_path)
    else:
        log.warning("Voice profile not found at %s -- skipping .bin conversion", args.profile)

    log.info("Done. Output files:")
    log.info("  ONNX model: %s", onnx_path)
    log.info("  Fbank config: %s", fbank_config_path)
    if os.path.exists(bin_path):
        log.info("  Voice profile: %s", bin_path)


if __name__ == "__main__":
    main()
