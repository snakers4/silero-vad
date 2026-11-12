import wave

import pytest
import torch

from silero_vad import VADIterator, load_silero_vad

from conftest import WAV

torch.set_num_threads(1)


def _read_wav(path):
    # test.wav is mono 16 kHz int16, read with the stdlib to avoid an audio backend
    with wave.open(path) as f:
        frames = f.readframes(f.getnframes())
    return torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768


@pytest.mark.parametrize("onnx", [False, True])
@pytest.mark.parametrize("threshold", [0.5, 0.15, 0.1])
def test_vad_iterator_ends_speech_at_low_threshold(onnx, threshold):
    # threshold - 0.15 is <= 0 here, which no probability can drop below
    vad_iterator = VADIterator(load_silero_vad(onnx=onnx), threshold=threshold)
    audio = _read_wav(WAV)

    events = []
    for i in range(0, len(audio) - 511, 512):
        speech_dict = vad_iterator(audio[i:i + 512])
        if speech_dict:
            events.append('start' if 'start' in speech_dict else 'end')

    assert 'end' in events
    assert events[::2] == ['start'] * len(events[::2])
    assert events[1::2] == ['end'] * len(events[1::2])


@pytest.mark.parametrize("onnx", [False, True])
@pytest.mark.parametrize("speech_pad_ms,min_silence_duration_ms", [(30, 100), (100, 100), (200, 300), (400, 500)])
def test_vad_iterator_segments_do_not_overlap(onnx, speech_pad_ms, min_silence_duration_ms):
    # with speech_pad_ms > min_silence_duration_ms / 2 a pause can be shorter
    # than both pads together; the next start must still not precede the last end
    vad_iterator = VADIterator(load_silero_vad(onnx=onnx), speech_pad_ms=speech_pad_ms,
                               min_silence_duration_ms=min_silence_duration_ms)
    audio = _read_wav(WAV)

    starts, ends = [], []
    for i in range(0, len(audio) - 511, 512):
        speech_dict = vad_iterator(audio[i:i + 512])
        if speech_dict and 'start' in speech_dict:
            starts.append(speech_dict['start'])
        elif speech_dict:
            ends.append(speech_dict['end'])

    assert len(ends) > 1
    assert len(starts) in (len(ends), len(ends) + 1)
    for end, next_start in zip(ends, starts[1:]):
        assert next_start >= end
