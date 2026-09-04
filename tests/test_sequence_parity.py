"""Parity test: sequence VAD path vs stock streaming get_speech_timestamps.

The sequence ONNX model is bit-exact with the stock model at the probability
level, so the derived speech timestamps must match exactly across a range of
post-processing parameters.

Run directly:  python tests/test_sequence_parity.py
Or via pytest: pytest tests/test_sequence_parity.py
"""
import os
import wave

import numpy as np
import pytest

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
    get_speech_timestamps_sequence,
)


def read_audio(path, sampling_rate=16000):
    """Read an uncompressed mono PCM16 WAV as float32 numpy at sampling_rate."""
    with wave.open(str(path), "rb") as source:
        if (source.getframerate() != sampling_rate
                or source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getcomptype() != "NONE"):
            raise ValueError(
                f"{path} must be uncompressed {sampling_rate} Hz mono PCM16 "
                f"(got {source.getframerate()} Hz, {source.getnchannels()} ch, "
                f"{source.getsampwidth()*8}-bit, {source.getcomptype()})"
            )
        raw = source.readframes(source.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WAVS = [
    os.path.join(REPO, "examples", "c++", "aepyx.wav"),
    os.path.join(HERE, "data", "test.wav"),
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


def _load_models():
    stock = load_silero_vad()  # jit streaming model
    seq = load_silero_vad(sequence=True)
    return stock, seq


def _cases():
    for wav in WAVS:
        if not os.path.exists(wav):
            continue
        for params in PARAM_SETS:
            yield wav, params


@pytest.mark.parametrize("wav,params", list(_cases()))
def test_parity(wav, params):
    stock, seq = _load_models()
    try:
        audio = read_audio(wav, sampling_rate=16000)
    except ValueError as exc:
        pytest.skip(str(exc))
    expected = get_speech_timestamps(audio, stock, sampling_rate=16000, **params)
    actual = get_speech_timestamps_sequence(audio, seq, sampling_rate=16000, **params)
    assert actual == expected, f"mismatch for {os.path.basename(wav)} {params}\n{actual}\n!=\n{expected}"


def main():
    stock, seq = _load_models()
    total = 0
    failures = 0
    for wav, params in _cases():
        try:
            audio = read_audio(wav, sampling_rate=16000)
        except ValueError as exc:
            print(f"[SKIP] {os.path.basename(wav):16s} {exc}")
            continue
        expected = get_speech_timestamps(audio, stock, sampling_rate=16000, **params)
        actual = get_speech_timestamps_sequence(audio, seq, sampling_rate=16000, **params)
        total += 1
        ok = actual == expected
        if not ok:
            failures += 1
        print(f"[{'OK' if ok else 'FAIL'}] {os.path.basename(wav):16s} "
              f"n_stock={len(expected):3d} n_seq={len(actual):3d} params={params}")
        if not ok:
            print("   expected:", expected)
            print("   actual:  ", actual)
    print(f"\n{total - failures}/{total} cases matched exactly.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
