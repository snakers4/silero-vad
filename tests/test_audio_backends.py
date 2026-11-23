"""Equivalence of the two audio I/O backends.

read_audio works through torchaudio when it is installed and through torchcodec
otherwise. The two are not bit-identical on files that need resampling (FFmpeg's
resampler differs from torchaudio.transforms.Resample), so what has to hold is
that the VAD sees the same speech either way.
"""
import sys

import pytest
import torch

from conftest import AUDIO_PATHS, MP3, WAV, requires_torchcodec

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

FRAME_SAMPLES = 512  # one model frame at 16 kHz


@pytest.fixture(scope="module")
def model():
    return load_silero_vad()


def _torchaudio_at_least_29():
    """Return torchaudio if it is installed and >= 2.9, else skip the test."""
    torchaudio = pytest.importorskip("torchaudio")
    ver = tuple(int(p) for p in torchaudio.__version__.split("+")[0].split(".")[:2])
    if ver < (2, 9):
        pytest.skip("the torchcodec fallback only exists for torchaudio >= 2.9")
    return torchaudio


def _boom(*args, **kwargs):
    raise RuntimeError("simulated torchaudio backend failure")


@requires_torchcodec
@pytest.mark.parametrize("path", AUDIO_PATHS)
def test_backends_agree_on_speech_timestamps(path, model, monkeypatch):
    """Both backends must yield the same speech segments.

    The samples themselves differ slightly on files that need resampling, so
    compare what actually matters: the timestamps the VAD derives from them.
    A frame whose probability sits right at the threshold can land on either
    side of it, so each boundary may move by one frame.
    """
    pytest.importorskip("torchaudio")
    expected = read_audio(path, sampling_rate=16000)
    expected_ts = get_speech_timestamps(expected, model)

    monkeypatch.setitem(sys.modules, "torchaudio", None)
    actual = read_audio(path, sampling_rate=16000)
    actual_ts = get_speech_timestamps(actual, model)

    assert actual.dim() == 1 and actual.dtype == torch.float32
    assert len(actual_ts) == len(expected_ts), f"{actual_ts} != {expected_ts}"
    for a, e in zip(actual_ts, expected_ts):
        assert abs(a['start'] - e['start']) <= FRAME_SAMPLES, f"{actual_ts} != {expected_ts}"
        assert abs(a['end'] - e['end']) <= FRAME_SAMPLES, f"{actual_ts} != {expected_ts}"


@requires_torchcodec
def test_backends_are_identical_without_resampling(monkeypatch):
    """test.wav is mono 16 kHz, so no resampling is involved at all and the two
    decoders have to agree exactly."""
    pytest.importorskip("torchaudio")
    expected = read_audio(WAV, sampling_rate=16000)

    monkeypatch.setitem(sys.modules, "torchaudio", None)
    assert torch.equal(read_audio(WAV, sampling_rate=16000), expected)


@requires_torchcodec
@pytest.mark.parametrize("path", AUDIO_PATHS)
def test_torchaudio_loader_failure_falls_back_to_torchcodec(path, monkeypatch):
    """torchaudio >= 2.9 delegates decoding to torchcodec; when its loader
    fails, read_audio must fall through and still return the same audio.

    This fallback decodes at the native rate and lets torchaudio resample, so
    unlike the torchaudio-less path it has to match bit for bit.
    """
    torchaudio = _torchaudio_at_least_29()
    expected = read_audio(path, sampling_rate=16000)

    monkeypatch.setattr(torchaudio, "load", _boom)
    assert torch.equal(read_audio(path, sampling_rate=16000), expected)


def test_torchaudio_without_torchcodec_reports_the_pin(without_torchcodec, monkeypatch):
    """torchaudio >= 2.9 with no torchcodec must explain the version pin."""
    torchaudio = _torchaudio_at_least_29()

    monkeypatch.setattr(torchaudio, "load", _boom)
    with pytest.raises(RuntimeError) as excinfo:
        read_audio(MP3, sampling_rate=16000)
    assert "torchcodec" in str(excinfo.value)


@requires_torchcodec
def test_stereo_is_averaged_not_ffmpeg_downmixed(without_torchaudio):
    """The torchcodec path must average the channels itself.

    Letting FFmpeg down-mix instead (AudioDecoder(num_channels=1)) produces a
    noticeably different signal - measured max delta 0.34 on this stereo mp3 -
    so guard against someone "simplifying" the decode call.
    """
    from torchcodec.decoders import AudioDecoder

    ours = read_audio(MP3, sampling_rate=16000)
    ffmpeg_mixed = AudioDecoder(MP3, sample_rate=16000,
                                num_channels=1).get_all_samples().data.squeeze(0)

    n = min(ours.numel(), ffmpeg_mixed.numel())
    assert not torch.allclose(ours[:n], ffmpeg_mixed[:n], atol=1e-3)
