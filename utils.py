import torch
import tempfile
import torchaudio
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from itertools import repeat

torchaudio.set_audio_backend("soundfile")  # switch backend


def read_batch(audio_paths: List[str]):
    return [read_audio(audio_path)
            for audio_path
            in audio_paths]


def split_into_batches(lst: List[str],
                       batch_size: int = 10):
    return [lst[i:i + batch_size]
            for i in
            range(0, len(lst), batch_size)]


def read_audio(path: str,
               target_sr: int = 16000):

    assert torchaudio.get_audio_backend() == 'soundfile'
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=target_sr)
        wav = transform(wav)
        sr = target_sr

    assert sr == target_sr
    return wav.squeeze(0)


def prepare_model_input(batch: List[torch.Tensor],
                        device=torch.device('cpu')):
    max_seqlength = max(max([len(_) for _ in batch]), 12800)
    inputs = torch.zeros(len(batch), max_seqlength)
    for i, wav in enumerate(batch):
        inputs[i, :len(wav)].copy_(wav)
    inputs = inputs.to(device)
    return inputs


#def init_jit_model(model_url: str,
#                   device: torch.device = torch.device('cpu')):
#    torch.set_grad_enabled(False)
#    with tempfile.NamedTemporaryFile('wb', suffix='.model') as f:
#        torch.hub.download_url_to_file(model_url,
#                                       f.name,
#                                       progress=True)
#        model = torch.jit.load(f.name, map_location=device)
#        model.eval()
#    return model


def init_jit_model(model_path,
                   device):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def get_speech_ts(wav, model, extractor, trig_sum=0.25, neg_trig_sum=0.01, num_steps=8, batch_size=200):
    assert 4000 % num_steps == 0
    step = int(4000 / num_steps)
    outs = []
    to_concat = []
    for i in range(0, len(wav), step):
        chunk = wav[i: i+4000]
        if len(chunk) < 4000:
            chunk = F.pad(chunk, (0, 4000 - len(chunk)))
        to_concat.append(chunk)
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.vstack(to_concat))
            with torch.no_grad():
                out = model(extractor(chunks))[-2]
            outs.append(out)
            to_concat = []
    
    if to_concat:
        chunks = torch.Tensor(torch.vstack(to_concat))
        with torch.no_grad():
            out = model(extractor(chunks))[-2]
        outs.append(out)
    
    outs = torch.cat(outs, dim=0)
    
    buffer = deque(maxlen=num_steps)
    triggered = False
    speeches = []
    current_speech = {}
    for i, predict in enumerate(outs[:, 1]):
        buffer.append(predict)
        if (np.mean(buffer) >= trig_sum) and not triggered:
            triggered = True
            current_speech['start'] = step * max(0, i-num_steps)
        if (np.mean(buffer) < neg_trig_sum) and triggered:
            current_speech['end'] = step * i
            if (current_speech['end'] - current_speech['start']) > 10000:
                speeches.append(current_speech)
            current_speech = {}
            triggered = False
    if current_speech:
        current_speech['end'] = len(wav)
        speeches.append(current_speech)
    return speeches


class STFTExtractor(nn.Module):
    def __init__(self, sr=16000, win_size=0.02, mode='mag'):
        super(STFTExtractor, self).__init__()
        self.sr = sr
        self.n_fft = int(sr * (win_size + 1e-8))
        self.win_length = self.n_fft
        self.hop_length = self.win_length // 2
        self.mode = 'mag' if mode == '' else mode

    def forward(self, wav):
        # center==True because other frame-level features are centered by default in torch/librosa and we can't change this.
        stft_sample = torch.stft(wav,
                                 n_fft=self.n_fft,
                                 win_length=self.win_length,
                                 hop_length=self.hop_length,
                                 center=True)
        mag, phase = torchaudio.functional.magphase(stft_sample)

        # It seems it is not a "mag", it is "power" (exp == 1).
        # Also there is "energy" (exp == 2).
        if self.mode == 'mag':
            return mag
        if self.mode == 'phase':
            return phase
        elif self.mode == 'magphase':
            return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)
        else:
            raise NotImplementedError()


class VADiterator:
    def __init__(self, trig_sum=0.26, neg_trig_sum=0.01, num_steps=8):
        self.num_steps = num_steps
        assert 4000 % num_steps == 0
        self.step = int(4000 / num_steps)
        self.prev = torch.zeros(4000)
        self.last = False
        self.triggered = False
        self.buffer = deque(maxlen=8)
        self.num_frames = 0
        self.trig_sum = trig_sum
        self.neg_trig_sum = neg_trig_sum
        self.current_name = ''
    
    def refresh(self):
        self.prev = torch.zeros(4000)
        self.last = False
        self.triggered = False
        self.buffer = deque(maxlen=8)
        self.num_frames = 0
    
    def prepare_batch(self, wav_chunk, name=None):
        if (name is not None) and (name != self.current_name):
            self.refresh()
            self.current_name = name
        assert len(wav_chunk) <= 4000
        self.num_frames += len(wav_chunk)
        if len(wav_chunk) < 4000:
            wav_chunk = F.pad(wav_chunk, (0, 4000 - len(wav_chunk))) # assume that short chunk means end of the audio
            self.last = True
        
        stacked = torch.hstack([self.prev, wav_chunk])
        self.prev = wav_chunk
        
        overlap_chunks = [stacked[i:i+4000] for i in range(500, 4001, self.step)] # 500 step is good enough
        return torch.vstack(overlap_chunks)
    
    def state(self, model_out):
        current_speech = {}
        for i, predict in enumerate(model_out[:, 1]):
            self.buffer.append(predict)
            if (np.mean(self.buffer) >= self.trig_sum) and not self.triggered:
                self.triggered = True
                current_speech[self.num_frames - (self.num_steps-i) * self.step] = 'start'
            if (np.mean(self.buffer) < self.neg_trig_sum) and self.triggered:
                current_speech[self.num_frames - (self.num_steps-i) * self.step] = 'end'
                self.triggered = False
        if self.triggered and self.last:
            current_speech[self.num_frames] = 'end'
        if self.last:
            self.refresh()
        return current_speech, self.current_name



def state_generator(model, audios, extractor, onnx=False, trig_sum=0.26, neg_trig_sum=0.01, num_steps=8, audios_in_stream=5):
    VADiters = [VADiterator(trig_sum, neg_trig_sum, num_steps) for i in range(audios_in_stream)]
    for i, current_pieces in enumerate(stream_imitator(audios, audios_in_stream)):
        for_batch = [x.prepare_batch(*y) for x, y in zip(VADiters, current_pieces)]
        batch = torch.cat(for_batch)

        with torch.no_grad():
            if onnx:
                ort_inputs = {'input': to_numpy(extractor(batch))}
                ort_outs = model.run(None, ort_inputs)
                vad_outs = np.split(ort_outs[-2], audios_in_stream)
            else:
                outs = model(extractor(batch))
                vad_outs = np.split(outs[-2].numpy(), audios_in_stream)
                
        states = []
        for x, y in zip(VADiters, vad_outs):
            cur_st = x.state(y)
            if cur_st[0]:
                states.append(cur_st)
        yield states


def stream_imitator(stereo, audios_in_stream):
    stereo_iter = iter(stereo)
    iterators = []
    # initial wavs
    for i in range(audios_in_stream):
        next_wav = next(stereo_iter)
        wav = read_audio(next_wav)
        wav_chunks = iter([(wav[i:i+4000], next_wav) for i in range(0, len(wav), 4000)])
        iterators.append(wav_chunks)
    print('Done initial Loading')
    good_iters = audios_in_stream
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                out, wav_name = next(it)
            except StopIteration:
                try:
                    next_wav = next(stereo_iter)
                    print('Loading next wav: ', next_wav)
                    wav = read_audio(next_wav)
                    iterators[i] = iter([(wav[i:i+4000], next_wav) for i in range(0, len(wav), 4000)])
                    out, wav_name = next(iterators[i])
                except StopIteration:
                    good_iters -= 1
                    iterators[i] = repeat((torch.zeros(4000), 'junk'))
                    out, wav_name = next(iterators[i])
                    if good_iters == 0:
                        return
            values.append((out, wav_name))
        yield values


