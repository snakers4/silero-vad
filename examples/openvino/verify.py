#!/usr/bin/env python3
"""Verify the converted model against the stock silero_vad.onnx.

Runs a chained state comparison on synthetic audio in both onnxruntime
and OpenVINO, and optionally on real wav files (16 bit mono PCM at the
matching sample rate). The converted model should be bit exact against
the stock one in onnxruntime, and within float rounding noise in
OpenVINO, with identical speech segmentation.

Usage:
    python verify.py converted.onnx [--sr 16000] [--stock path] [wav ...]

Exit code is nonzero if the onnxruntime comparison is not bit exact, if
the OpenVINO max abs diff exceeds 1e-4, or if segmentation differs.

Requires: numpy, onnx, onnxruntime, openvino.
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import openvino as ov

REPO_MODEL = Path(__file__).resolve().parents[2] / \
    'src' / 'silero_vad' / 'data' / 'silero_vad.onnx'


def synthetic_audio(sr, rng):
    """A varied stretch of synthetic audio. The comparison is numeric,
    it does not need to be realistic speech."""
    t = lambda sec: np.arange(int(sec * sr)) / sr
    parts = [np.zeros(int(3 * sr), dtype=np.float32)]
    tt = t(4)  # mains hum with harmonics
    parts.append((0.02 * np.sin(2 * np.pi * 60 * tt)
                  + 0.01 * np.sin(2 * np.pi * 120 * tt)
                  + 0.005 * np.sin(2 * np.pi * 180 * tt)).astype(np.float32))
    parts.append((0.05 * rng.standard_normal(int(3 * sr))).astype(np.float32))
    tt = t(4)  # amplitude modulated noise with formant like tones
    env = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 4 * tt)))
    carrier = np.sin(2 * np.pi * 220 * tt) + 0.6 * np.sin(2 * np.pi * 710 * tt) \
        + 0.3 * np.sin(2 * np.pi * 2400 * tt)
    parts.append((0.15 * env * carrier
                  + 0.02 * rng.standard_normal(len(tt))).astype(np.float32))
    tt = t(3)
    parts.append((0.1 * np.sin(2 * np.pi * (100 + 900 * tt) * tt)).astype(np.float32))
    parts.append((0.3 * rng.standard_normal(int(3 * sr))).astype(np.float32))
    parts.append(np.zeros(int(2 * sr), dtype=np.float32))
    return np.concatenate(parts)


def load_wav(path, sr):
    w = wave.open(str(path))
    assert w.getframerate() == sr and w.getnchannels() == 1 \
        and w.getsampwidth() == 2, f"{path}: expected {sr} Hz mono PCM16"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


class OrtStock:
    def __init__(self, path, sr):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s = ort.InferenceSession(str(path), so,
                                      providers=['CPUExecutionProvider'])
        self.sr = np.array(sr, dtype=np.int64)
        self.ctx_size = 64 if sr == 16000 else 32

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, self.ctx_size), np.float32)

    def __call__(self, chunk):
        x = np.concatenate([self.ctx, chunk[None]], axis=1)
        p, self.state = self.s.run(
            ['output', 'stateN'],
            {'input': x, 'state': self.state, 'sr': self.sr})
        self.ctx = x[:, -self.ctx_size:]
        return float(p[0, 0])


class OrtConverted(OrtStock):
    def __call__(self, chunk):
        x = np.concatenate([self.ctx, chunk[None]], axis=1)
        p, self.state = self.s.run(['output', 'stateN'],
                                   {'input': x, 'state': self.state})
        self.ctx = x[:, -self.ctx_size:]
        return float(p[0, 0])


class OvConverted:
    def __init__(self, path, sr):
        core = ov.Core()
        # Force f32 inference. On bf16 capable CPUs the OpenVINO CPU plugin
        # defaults to bf16, and the per step error compounds through the
        # recurrent state until the speech segmentation changes.
        compiled = core.compile_model(str(path), 'CPU',
                                      {'INFERENCE_PRECISION_HINT': 'f32'})
        self.r = compiled.create_infer_request()
        self.ctx_size = 64 if sr == 16000 else 32

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, self.ctx_size), np.float32)

    def __call__(self, chunk):
        x = np.concatenate([self.ctx, chunk[None]], axis=1)
        res = self.r.infer({'input': x, 'state': self.state})
        self.state = res['stateN']
        self.ctx = x[:, -self.ctx_size:]
        return float(res['output'][0, 0])


def segments(probs, thr=0.5, min_chunks=8):
    segs, start = [], None
    for i, p in enumerate(probs):
        if p >= thr and start is None:
            start = i
        elif p < thr and start is not None:
            if i - start >= min_chunks:
                segs.append((start, i))
            start = None
    if start is not None and len(probs) - start >= min_chunks:
        segs.append((start, len(probs)))
    return segs


def run(audio, chunk, engines):
    for e in engines:
        e.reset()
    n = len(audio) // chunk
    out = [[] for _ in engines]
    for i in range(n):
        c = audio[i * chunk:(i + 1) * chunk]
        for j, e in enumerate(engines):
            out[j].append(e(c))
    return [np.array(o) for o in out]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('converted', help='path to the converted onnx')
    p.add_argument('wavs', nargs='*', help='optional real wav files')
    p.add_argument('--sr', type=int, choices=[8000, 16000], default=16000)
    p.add_argument('--stock', default=str(REPO_MODEL),
                   help='path to the stock silero_vad.onnx')
    a = p.parse_intermixed_args()

    chunk = 512 if a.sr == 16000 else 256
    stock = OrtStock(a.stock, a.sr)
    conv_ort = OrtConverted(a.converted, a.sr)
    conv_ov = OvConverted(a.converted, a.sr)
    print(f"sr={a.sr}, chunk={chunk}, OpenVINO {ov.__version__}, inference precision forced to f32")

    ok = True
    rng = np.random.default_rng(42)
    audio = synthetic_audio(a.sr, rng)
    ps, po, pv = run(audio, chunk, [stock, conv_ort, conv_ov])
    d_ort = np.abs(po - ps).max()
    d_ov = np.abs(pv - ps).max()
    print(f"\nsynthetic ({len(audio)//chunk} chunks, chained state)")
    print(f"  onnxruntime converted vs stock: max abs diff = {d_ort:.3e}"
          f"  {'(bit exact)' if d_ort == 0 else 'EXPECTED 0.0'}")
    print(f"  OpenVINO converted vs stock:    max abs diff = {d_ov:.3e}")
    ok &= (d_ort == 0.0) and (d_ov < 1e-4)

    for wav in a.wavs:
        audio = load_wav(wav, a.sr)
        ps, pv = run(audio, chunk, [stock, conv_ov])
        d = np.abs(pv - ps).max()
        s_ref, s_ov = segments(ps), segments(pv)
        same = s_ref == s_ov
        print(f"\n{wav} ({len(audio)/a.sr:.0f} s, {len(ps)} chunks)")
        print(f"  OpenVINO vs stock: max abs diff = {d:.3e}")
        print(f"  speech segments: stock={len(s_ref)} OpenVINO={len(s_ov)}"
              f"  identical={same}")
        ok &= same and (d < 1e-4)

    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
