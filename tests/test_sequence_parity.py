"""Parity test: sequence VAD path vs stock streaming get_speech_timestamps.

The sequence ONNX model is bit-exact with the stock model at the probability
level, so the derived speech timestamps must match exactly across a range of
post-processing parameters.

Run with: pytest tests/test_sequence_parity.py
"""
import functools
import os

import numpy as np
import pytest
import soundfile as sf

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
    get_speech_timestamps_sequence,
)


def read_audio(path, sampling_rate=16000):
    """Read any soundfile-supported audio as mono float32 at sampling_rate.

    Parity only requires that both the stock and sequence paths receive the
    exact same samples, so a simple deterministic linear resample is sufficient
    here (audio quality is irrelevant to the equality check).
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:  # down-mix to mono
        audio = audio.mean(axis=1).astype(np.float32)
    if sr != sampling_rate:
        n_out = int(round(audio.size * sampling_rate / sr))
        x = np.linspace(0.0, 1.0, audio.size, endpoint=False, dtype=np.float64)
        xn = np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float64)
        audio = np.interp(xn, x, audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
# Self-contained test audio shipped under tests/data/ (wav/opus/mp3).
WAVS = [
    os.path.join(DATA_DIR, "test.wav"),
    os.path.join(DATA_DIR, "test.opus"),
    os.path.join(DATA_DIR, "test.mp3"),
]

PARAM_SETS = [
    dict(),
    dict(threshold=0.3),
    dict(return_seconds=True),
    dict(return_seconds=True, time_resolution=2),
    dict(min_speech_duration_ms=100, min_silence_duration_ms=50),
    dict(speech_pad_ms=0),
    dict(speech_pad_ms=100),
    dict(max_speech_duration_s=5.0),
    dict(max_speech_duration_s=5.0, use_max_poss_sil_at_max_speech=False),
    dict(max_speech_duration_s=3.0, min_silence_at_max_speech=50),
]


@functools.lru_cache(maxsize=1)
def _load_models():
    # Cached so the models are loaded once and reused across all parametrized
    # cases instead of being reloaded for each.
    stock = load_silero_vad()  # jit streaming model
    seq = load_silero_vad(sequence=True)
    return stock, seq


def _cases():
    for wav in WAVS:
        for params in PARAM_SETS:
            yield wav, params


@pytest.mark.parametrize("wav,params", list(_cases()))
def test_parity(wav, params):
    stock, seq = _load_models()
    assert os.path.exists(wav), f"missing test audio: {wav}"
    audio = read_audio(wav, sampling_rate=16000)
    expected = get_speech_timestamps(audio, stock, sampling_rate=16000, **params)
    actual = get_speech_timestamps_sequence(audio, seq, sampling_rate=16000, **params)
    assert actual == expected, f"mismatch for {os.path.basename(wav)} {params}\n{actual}\n!=\n{expected}"
