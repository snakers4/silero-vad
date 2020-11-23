dependencies = ['torch', 'omegaconf', 'torchaudio']
import torch
from omegaconf import OmegaConf
from utils import (init_jit_model,
                   read_audio,
                   read_batch,
                   split_into_batches,
                   prepare_model_input)


def silero_stt(**kwargs):
    """Silero Voice Activity and Number Detector Models
    Returns a model and a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-vad/master/models.yml',
                                   'silero_vad_models.yml',
                                   progress=False)
    models = OmegaConf.load('silero_vad_models.yml')

    model = init_jit_model(model_url=models.latest.jit,
                           **kwargs)
    utils = (read_batch,
             split_into_batches,
             read_audio,
             prepare_model_input)

    return model, utils
