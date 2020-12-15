dependencies = ['torch', 'torchaudio']
import torch
from utils import (init_jit_model,
                   get_speech_ts,
                   save_audio,
                   read_audio,
                   state_generator,
                   single_audio_stream,
                   collect_speeches)


def silero_vad(**kwargs):
    """Silero Voice Activity Detector, Number Detector and Language Classifier
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    model = init_jit_model(model_path='files/model.jit')
    utils = (get_speech_ts,
             save_audio,
             read_audio,
             state_generator,
             single_audio_stream,
             collect_speeches)

    return model, utils
