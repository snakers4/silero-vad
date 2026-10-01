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
@pytest.mark.parametrize("sampling_rate", [8000, 16000])
@pytest.mark.parametrize("batched,return_seconds", [(False, False), (True, True)])
def test_vad_iterator_rejected_chunk_does_not_advance_stream(
        onnx, sampling_rate, batched, return_seconds):
    reference = VADIterator(load_silero_vad(onnx=onnx), sampling_rate=sampling_rate)
    recovering = VADIterator(load_silero_vad(onnx=onnx), sampling_rate=sampling_rate)
    audio = _read_wav(WAV)[:16000 * 5][::16000 // sampling_rate]
    audio = torch.cat((audio, torch.zeros(sampling_rate)))
    window_size = 512 if sampling_rate == 16000 else 256
    rejected_phases = set()
    events = []

    for i in range(0, len(audio) - window_size + 1, window_size):
        phase = 'pending_end' if recovering.temp_end else (
            'speech' if recovering.triggered else 'idle')
        if phase not in rejected_phases:
            invalid = torch.zeros(2 * window_size)
            if batched:
                invalid = invalid.unsqueeze(0)
            before = (recovering.current_sample, recovering.triggered, recovering.temp_end)
            with pytest.raises((ValueError, torch.jit.Error), match="Provided number of samples"):
                recovering(invalid)
            assert (recovering.current_sample, recovering.triggered, recovering.temp_end) == before
            rejected_phases.add(phase)

        chunk = audio[i:i + window_size]
        if batched:
            chunk = chunk.unsqueeze(0)
        expected = reference(chunk, return_seconds=return_seconds, time_resolution=3)
        actual = recovering(chunk, return_seconds=return_seconds, time_resolution=3)
        assert actual == expected
        if actual:
            events.append('start' if 'start' in actual else 'end')

    assert rejected_phases == {'idle', 'speech', 'pending_end'}
    assert 'start' in events
    assert 'end' in events
    assert recovering.current_sample == reference.current_sample
