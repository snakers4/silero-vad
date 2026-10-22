

# Example to remove human voices from the wav files using silero-vad and process them in parallel using ProcessPoolExecutor


import os
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint
import time

SR = 16000
NUM_PROCESS = 1  # set to the number of CPU cores in the machine

torch.set_num_threads(1)
wav_dir = './sdata/'

# A wrapper class to make the model pickleable
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        return self.model.state_dict()

    def __setstate__(self, state):
        model, utils = get_model_and_utils()
        model.load_state_dict(state)
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __repr__(self):
        return repr(self.model)

    def __str__(self):
        return str(self.model)

def drop_chunks(tss, wav):
    if len(tss) == 0:
        return wav
    chunks = []
    cur_start = 0
    for i in tss:
        chunks.append((wav[cur_start: i['start']]))
        cur_start = i['end']
    return torch.cat(chunks)


def get_model_and_utils():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                   model='silero_vad',
                                   force_reload=False,
                                   onnx=False)
    model = ModelWrapper(model)
    return model, utils


def process_wav_file(wav_file: str, model, utils):
    print("Processing file: " + wav_file)
    model, utils = get_model_and_utils()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    label_name = determine_label_name(wav_file)
    wav = load_wav_file(wav_file)

    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SR)
    final_wav = drop_chunks(speech_timestamps, wav)
    torchaudio.save(f'./sdata1/{wav_file}', final_wav.unsqueeze(0), SR)
    wav = final_wav[SR:-SR]

    return wav


def determine_label_name(wav_file):
    # A function to process label name from wav file name
    return 'label'


def load_wav_file(wav_file):
    # Load wav and resample if necessary
    wav, sample_rate = torchaudio.load(wav_dir + wav_file)
    wav = wav.mean(dim=0) if wav.ndim > 1 else wav
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        wav = resample_transform(wav)
    return wav

def initialize_vad(model, utils):
    global vad_model, vad_utils
    vad_model = model
    vad_utils = utils

def worker_function(wav_file):
    global vad_model, vad_utils
    return process_wav_file(wav_file, vad_model, vad_utils)

def main():
    futures = []
    data = []

    model, utils = get_model_and_utils()

    with ProcessPoolExecutor(max_workers=NUM_PROCESS, initializer=initialize_vad, initargs=(model, utils)) as ex:
        wav_files = sorted(os.listdir(wav_dir))
        for wav_file in wav_files:
            futures.append(ex.submit(worker_function, wav_file))


    for finished in as_completed(futures):
        result = finished.result()
        data.extend(result)

    pprint(data)
    pprint(len(data))

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    pprint(f"Execution time: {end_time - start_time:.4f} seconds")
