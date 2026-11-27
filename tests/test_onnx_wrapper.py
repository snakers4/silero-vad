import os
import shutil
import warnings

import pytest
import torch

import silero_vad
from silero_vad import get_speech_timestamps
from silero_vad.utils_vad import OnnxWrapper

torch.set_num_threads(1)

MODEL_DIR = os.path.join(os.path.dirname(silero_vad.__file__), "data")


def test_16k_in_a_parent_directory_does_not_restrict_sample_rates(tmp_path):
    # e.g. a virtualenv that lives in .../asr_16k_project/
    model_dir = tmp_path / "asr_16k_project"
    model_dir.mkdir()
    path = str(model_dir / "silero_vad.onnx")
    shutil.copy(os.path.join(MODEL_DIR, "silero_vad.onnx"), path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = OnnxWrapper(path, force_onnx_cpu=True)

    assert model.sample_rates == [8000, 16000]
    assert get_speech_timestamps(torch.zeros(8000), model, sampling_rate=8000) == []


def test_16k_only_model_is_still_detected_by_file_name():
    with pytest.warns(UserWarning, match="only 16000"):
        model = OnnxWrapper(os.path.join(MODEL_DIR, "silero_vad_16k_op15.onnx"), force_onnx_cpu=True)

    assert model.sample_rates == [16000]
