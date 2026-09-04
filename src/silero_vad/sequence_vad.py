"""Batched ("sequence") ONNX front-end for Silero VAD.

The stock streaming model processes audio one frame at a time (512 samples for
16 kHz), carrying the LSTM state between frames in a Python loop. That per-frame
Python loop holds the GIL and dominates wall-time in multi-threaded servers,
even though the ONNX kernels themselves release the GIL.

This module runs the *same* computation as a single ONNX call per block of
frames (by default up to 512 frames ~= 16.4 s of 16 kHz audio), producing
bit-exact per-frame probabilities while spending almost all of its time inside
GIL-releasing ONNX Runtime kernels. The probabilities are then converted to
speech timestamps with the shared ``get_speech_timestamps_from_probs`` state
machine.

The module is intentionally torch-free (only numpy + onnxruntime) so it can be
imported and used on worker threads without pulling in torch.
"""

import numpy as np
import onnxruntime as ort

from .utils_vad import get_speech_timestamps_from_probs

# (frame_samples, context_samples) per supported sample rate.
RATE_CONFIG = {
    8000: (256, 32),
    16000: (512, 64),
}
STATE_SHAPE = (2, 1, 128)
# Default max frames per ONNX call. 512 frames ~= 16.4 s at 16 kHz.
DEFAULT_MAX_FRAMES = 512


def frame_blocks(audio: np.ndarray,
                 frame_samples: int,
                 context_samples: int,
                 max_frames: int):
    """Yield blocks of shape (block_frames, context_samples + frame_samples).

    Each frame is prefixed with the trailing ``context_samples`` of the previous
    frame (zeros for the very first frame), matching the streaming model's
    internal context handling. The final frame is zero-padded to a full frame.
    """
    frame_count = (audio.size + frame_samples - 1) // frame_samples
    previous_context = np.zeros(context_samples, dtype=np.float32)
    for first_frame in range(0, frame_count, max_frames):
        block_frames = min(max_frames, frame_count - first_frame)
        first_sample = first_frame * frame_samples
        samples = audio[
            first_sample: min(audio.size, (first_frame + block_frames) * frame_samples)
        ]
        frames = np.zeros((block_frames, frame_samples), dtype=np.float32)
        frames.reshape(-1)[: samples.size] = samples

        contexts = np.empty((block_frames, context_samples), dtype=np.float32)
        contexts[0] = previous_context
        if block_frames > 1:
            contexts[1:] = frames[:-1, -context_samples:]
        previous_context = frames[-1, -context_samples:].copy()
        yield np.concatenate((contexts, frames), axis=1)


def _to_numpy_1d(audio) -> np.ndarray:
    """Convert audio (numpy array, torch tensor, or sequence) to 1-D float32."""
    if isinstance(audio, np.ndarray):
        arr = audio
    elif hasattr(audio, "detach"):  # torch.Tensor without importing torch
        arr = audio.detach().cpu().numpy()
    elif hasattr(audio, "numpy"):
        arr = audio.numpy()
    else:
        arr = np.asarray(audio)
    arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    return arr


class SileroVADSequence:
    """Loads a sequence ONNX model and computes per-frame speech probabilities.

    Parameters
    ----------
    path: str
        Path to the sequence ONNX model.
    sampling_rate: int (default - 16000)
        Native sample rate of the exported model.
    max_frames: int (default - 512)
        Maximum frames per ONNX call. Larger values reduce Python/GIL overhead
        at the cost of more transient memory per call.
    force_onnx_cpu: bool (default - True)
        Force the CPU execution provider.
    """

    def __init__(self,
                 path: str,
                 sampling_rate: int = 16000,
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 force_onnx_cpu: bool = True):
        if sampling_rate not in RATE_CONFIG:
            raise ValueError(
                f"Unsupported sampling_rate {sampling_rate}; supported: {sorted(RATE_CONFIG)}"
            )
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"] if force_onnx_cpu else None
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=providers,
        )
        self.sampling_rate = sampling_rate
        self.max_frames = max_frames

    def audio_forward(self,
                      audio: np.ndarray,
                      sampling_rate: int = 16000,
                      max_frames: int = None) -> np.ndarray:
        """Return a 1-D float32 array of per-frame speech probabilities.

        Runs the model as a single ONNX call per block of up to ``max_frames``
        frames, so the Python loop iterates only ceil(num_frames / max_frames)
        times instead of once per frame.
        """
        if sampling_rate not in RATE_CONFIG:
            raise ValueError(
                f"Unsupported sampling_rate {sampling_rate}; supported: {sorted(RATE_CONFIG)}"
            )
        if max_frames is None:
            max_frames = self.max_frames
        frame_samples, context_samples = RATE_CONFIG[sampling_rate]

        hidden = np.zeros((1, 1, 128), dtype=np.float32)
        cell = np.zeros((1, 1, 128), dtype=np.float32)
        probabilities = []
        for block in frame_blocks(audio, frame_samples, context_samples, max_frames):
            values, hidden, cell = self.session.run(
                ["speech_probs", "hn", "cn"],
                {"input": block, "h": hidden, "c": cell},
            )
            probabilities.append(values)
        if not probabilities:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(probabilities).reshape(-1)


def get_speech_timestamps_sequence(audio,
                                   model: SileroVADSequence,
                                   threshold: float = 0.5,
                                   sampling_rate: int = 16000,
                                   min_speech_duration_ms: int = 250,
                                   max_speech_duration_s: float = float('inf'),
                                   min_silence_duration_ms: int = 100,
                                   speech_pad_ms: int = 30,
                                   return_seconds: bool = False,
                                   time_resolution: int = 1,
                                   visualize_probs: bool = False,
                                   neg_threshold: float = None,
                                   min_silence_at_max_speech: int = 98,
                                   use_max_poss_sil_at_max_speech: bool = True,
                                   max_frames: int = None):
    """Drop-in analogue of ``get_speech_timestamps`` using the sequence model.

    ``model`` must be a ``SileroVADSequence`` (e.g. from
    ``load_silero_vad(sequence=True)``). Produces the same speech timestamps as
    the stock streaming path but with the per-frame inference loop replaced by a
    handful of block-level ONNX calls.
    """
    audio = _to_numpy_1d(audio)

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
    else:
        step = 1

    if sampling_rate not in RATE_CONFIG:
        raise ValueError(
            "Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates"
        )

    audio_length_samples = audio.size
    speech_probs = model.audio_forward(audio, sampling_rate=sampling_rate, max_frames=max_frames)

    return get_speech_timestamps_from_probs(
        speech_probs,
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=return_seconds,
        time_resolution=time_resolution,
        visualize_probs=visualize_probs,
        neg_threshold=neg_threshold,
        min_silence_at_max_speech=min_silence_at_max_speech,
        use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech,
        audio_length_samples=audio_length_samples,
        step=step,
    )
