from utils_vad import *
import sys
import os
from pathlib import Path
sys.path.append('/home/keras/notebook/nvme_raid/adamnsandle/silero_mono/pipelines/align/bin/')
from align_utils import load_audio_norm
import torch
import pandas as pd
import numpy as np
sys.path.append('/home/keras/notebook/nvme_raid/adamnsandle/silero_mono/utils/')
from open_stt import soundfile_opus as sf

def split_save_audio_chunks(audio_path, model_path, save_path=None, device='cpu', absolute=True, max_duration=10, adaptive=False, **kwargs):

    if not save_path:
        save_path = str(Path(audio_path).with_name('after_vad'))
        print(f'No save path specified! Using {save_path} to save audio chunks!')

    SAMPLE_RATE = 16000
    if type(model_path) == str:
        #print('Loading model...')
        model = init_jit_model(model_path, device)
    else:
        #print('Using loaded model')
        model = model_path
    save_name = Path(audio_path).stem
    audio, sr = load_audio_norm(audio_path)
    wav = torch.tensor(audio)
    if adaptive:
        speech_timestamps = get_speech_ts_adaptive(wav, model, device=device, **kwargs)
    else:
        speech_timestamps = get_speech_ts(wav, model, device=device, **kwargs)

    full_save_path = Path(save_path, save_name)
    if not os.path.exists(full_save_path):
        os.makedirs(full_save_path, exist_ok=True)

    chunks = []
    if not speech_timestamps:
        return pd.DataFrame()
    for ts in speech_timestamps:
        start_ts = int(ts['start'])
        end_ts = int(ts['end'])

        for i in range(start_ts, end_ts, max_duration * SAMPLE_RATE):
            new_start = i
            new_end = min(end_ts, i + max_duration * SAMPLE_RATE)
            duration = round((new_end - new_start) / SAMPLE_RATE, 2)
            chunk_path = Path(full_save_path, f'{save_name}_{new_start}-{new_end}.opus')
            chunk_path = chunk_path.absolute() if absolute else chunk_path
            sf.write(str(chunk_path), audio[new_start: new_end], 16000, format='OGG', subtype='OPUS')
            chunks.append({'audio_path': chunk_path,
                        'text': '', 
                        'duration': duration,
                        'domain': ''})
    return pd.DataFrame(chunks)