from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks, VADIterator
import torch
torch.set_num_threads(1)

def test_jit_model():
    model = load_silero_vad(onnx=False)
    for path in ["tests/data/test.wav", "tests/data/test.opus", "tests/data/test.mp3"]:
        audio = read_audio(path, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(audio, model, visualize_probs=False, return_seconds=True)
        assert speech_timestamps is not None
        out = model.audio_forward(audio, sr=16000)
        assert out is not None

def test_onnx_model():
    model = load_silero_vad(onnx=True)
    for path in ["tests/data/test.wav", "tests/data/test.opus", "tests/data/test.mp3"]:
        audio = read_audio(path, sampling_rate=16000)
        speech_timestamps = get_speech_timestamps(audio, model, visualize_probs=False, return_seconds=True)
        assert speech_timestamps is not None

        out = model.audio_forward(audio, sr=16000)
        assert out is not None


def test_collect_chunks_without_speech():
    model = load_silero_vad(onnx=False)
    wav = torch.zeros(16000)
    speech_timestamps = get_speech_timestamps(wav, model)
    assert speech_timestamps == []
    out = collect_chunks(speech_timestamps, wav)
    assert out.shape == (0,)
    assert out.dtype == wav.dtype
    out = collect_chunks(speech_timestamps, wav, seconds=True, sampling_rate=16000)
    assert out.shape == (0,)


def test_float64_audio_gives_the_same_timestamps_as_float32():
    # torch.from_numpy(soundfile.read(path)[0]) is float64, which used to crash
    # both models instead of being handled like the float32 tensor it holds
    audio = read_audio("tests/data/test.wav", sampling_rate=16000)
    for onnx in [False, True]:
        model = load_silero_vad(onnx=onnx)
        expected = get_speech_timestamps(audio, model)
        assert expected
        assert get_speech_timestamps(audio.double(), model) == expected


def test_vad_iterator_accepts_float64_chunks():
    audio = read_audio("tests/data/test.wav", sampling_rate=16000)
    for onnx in [False, True]:
        model = load_silero_vad(onnx=onnx)
        events = {}
        for dtype in [torch.float32, torch.float64]:
            vad_iterator = VADIterator(model)
            events[dtype] = [vad_iterator(audio[i:i + 512].to(dtype))
                             for i in range(0, len(audio) - 511, 512)]
        assert any(events[torch.float32])
        assert events[torch.float64] == events[torch.float32]
