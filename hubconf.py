dependencies = ['torch', 'torchaudio']
import torch
import json
from utils_vad import (init_jit_model,
                       get_speech_timestamps,
                       get_number_ts,
                       get_language,
                       get_language_and_group,
                       save_audio,
                       read_audio,
                       VADIterator,
                       collect_chunks,
                       drop_chunks,
                       Validator,
                       OnnxWrapper)


def silero_vad(onnx=False):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    hub_dir = torch.hub.get_dir()
    if onnx:
        model = OnnxWrapper(f'{hub_dir}/snakers4_silero-vad_master/files/silero_vad.onnx')
    else:
        model = init_jit_model(model_path=f'{hub_dir}/snakers4_silero-vad_master/files/silero_vad.jit')
    utils = (get_speech_timestamps,
             save_audio,
             read_audio,
             VADIterator,
             collect_chunks)

    return model, utils


def silero_number_detector(onnx=False):
    """Silero Number Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    if onnx:
        url = 'https://models.silero.ai/vad_models/number_detector.onnx'
    else:
        url = 'https://models.silero.ai/vad_models/number_detector.jit'
    model = Validator(url)
    utils = (get_number_ts,
             save_audio,
             read_audio,
             collect_chunks,
             drop_chunks)

    return model, utils


def silero_lang_detector(onnx=False):
    """Silero Language Classifier
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    if onnx:
        url = 'https://models.silero.ai/vad_models/number_detector.onnx'
    else:
        url = 'https://models.silero.ai/vad_models/number_detector.jit'
    model = Validator(url)
    utils = (get_language,
             read_audio)

    return model, utils


def silero_lang_detector_95(onnx=False):
    """Silero Language Classifier (95 languages)
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """

    hub_dir = torch.hub.get_dir()
    if onnx:
        url = 'https://models.silero.ai/vad_models/lang_classifier_95.onnx'
    else:
        url = 'https://models.silero.ai/vad_models/lang_classifier_95.jit'
    model = Validator(url)

    with open(f'{hub_dir}/snakers4_silero-vad_master/files/lang_dict_95.json', 'r') as f:
        lang_dict = json.load(f)

    with open(f'{hub_dir}/snakers4_silero-vad_master/files/lang_group_dict_95.json', 'r') as f:
        lang_group_dict = json.load(f)

    utils = (get_language_and_group, read_audio)

    return model, lang_dict, lang_group_dict, utils
