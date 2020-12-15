dependencies = ['torch', 'torchaudio', 'numpy']
import torch
from omegaconf import OmegaConf
from utils import (init_jit_model, 
                   get_speech_ts,
                   save_audio, 
                   read_audio, 
                   state_generator, 
                   single_audio_stream,
                   collect_speeches)


def silero_stt(**kwargs):
    """Silero Voice Activity and Number Detector Models
    Returns a model and a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-vad/master/files/model.jit',
                                   'files/model.jit',
                                   progress=False)
    model = init_jit_model(model_url='files/model.jit')
    utils = (get_speech_ts,
             save_audio, 
             read_audio, 
             state_generator, 
             single_audio_stream,
             collect_speeches)

    return model, utils
