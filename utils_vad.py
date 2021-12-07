import torch
import torchaudio
from typing import List
import torch.nn.functional as F
import warnings

languages = ['ru', 'en', 'de', 'es']


def validate(model,
             inputs: torch.Tensor,
             **kwargs):
    with torch.no_grad():
        outs = model(inputs, **kwargs)
    if len(outs.shape) == 1:
        return outs[1:]
    return outs[:, 1]  # 0 for noise, 1 for speech


def read_audio(path: str,
               sampling_rate: int = 16000):

    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)


def save_audio(path: str,
               tensor: torch.Tensor,
               sampling_rate: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate)


def init_jit_model(model_path: str,
                   device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def make_visualization(probs, step):
    import pandas as pd
    pd.DataFrame({'probs': probs},
                 index=[x * step for x in range(len(probs))]).plot(figsize=(16, 8),
                 kind='area', ylim=[0, 1.05], xlim=[0, len(probs) * step],
                 xlabel='seconds',
                 ylabel='speech probability',
                 colormap='tab20')


def get_speech_timestamps(audio: torch.Tensor,
                          model,
                          threshold: float = 0.5,
                          sample_rate: int = 16000,
                          min_speech_duration_ms: int = 250,
                          min_silence_duration_ms: int = 100,
                          window_size_samples: int = 1536,
                          speech_pad_ms: int = 30,
                          return_seconds: bool = False,
                          visualize_probs: bool = False):

    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    model: preloaded .jit silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    sample_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model perfomance!!

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)

    visualize_probs: bool (default - False)
        whether draw prob hist or not

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

    if sample_rate == 8000 and window_size_samples > 768:
        warnings.warn('window_size_samples is too big for 8000 sample_rate! Better set window_size_samples to 256, 512 or 1536 for 8000 sample rate!')
    if window_size_samples not in [256, 512, 768, 1024, 1536]:
        warnings.warn('Unusual window_size_samples! Supported window_size_samples:\n - [512, 1024, 1536] for 16000 sample_rate\n - [256, 512, 768] for 8000 sample_rate')

    model.reset_states()
    min_speech_samples = sample_rate * min_speech_duration_ms / 1000
    min_silence_samples = sample_rate * min_silence_duration_ms / 1000
    speech_pad_samples = sample_rate * speech_pad_ms / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_prob = model(chunk, sample_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue

    if current_speech:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] += int(speech_pad_samples)
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        for speech_dict in speeches:
            speech_dict['start'] = round(speech_dict['start'] / sample_rate, 1)
            speech_dict['end'] = round(speech_dict['end'] / sample_rate, 1)

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sample_rate)

    return speeches


def get_number_ts(wav: torch.Tensor,
                  model,
                  model_stride=8,
                  hop_length=160,
                  sample_rate=16000,
                  run_function=validate):
    wav = torch.unsqueeze(wav, dim=0)
    perframe_logits = run_function(model, wav)[0]
    perframe_preds = torch.argmax(torch.softmax(perframe_logits, dim=1), dim=1).squeeze()   # (1, num_frames_strided)
    extended_preds = []
    for i in perframe_preds:
        extended_preds.extend([i.item()] * model_stride)
    # len(extended_preds) is *num_frames_real*; for each frame of audio we know if it has a number in it.
    triggered = False
    timings = []
    cur_timing = {}
    for i, pred in enumerate(extended_preds):
        if pred == 1:
            if not triggered:
                cur_timing['start'] = int((i * hop_length) / (sample_rate / 1000))
                triggered = True
        elif pred == 0:
            if triggered:
                cur_timing['end'] = int((i * hop_length) / (sample_rate / 1000))
                timings.append(cur_timing)
                cur_timing = {}
                triggered = False
    if cur_timing:
        cur_timing['end'] = int(len(wav) / (sample_rate / 1000))
        timings.append(cur_timing)
    return timings


def get_language(wav: torch.Tensor,
                 model,
                 run_function=validate):
    wav = torch.unsqueeze(wav, dim=0)
    lang_logits = run_function(model, wav)[2]
    lang_pred = torch.argmax(torch.softmax(lang_logits, dim=1), dim=1).item()   # from 0 to len(languages) - 1
    assert lang_pred < len(languages)
    return languages[lang_pred]


def get_language_and_group(wav: torch.Tensor,
                           model,
                           lang_dict: dict,
                           lang_group_dict: dict,
                           top_n=1,
                           run_function=validate):
    wav = torch.unsqueeze(wav, dim=0)
    lang_logits, lang_group_logits = run_function(model, wav)

    softm = torch.softmax(lang_logits, dim=1).squeeze()
    softm_group = torch.softmax(lang_group_logits, dim=1).squeeze()

    srtd = torch.argsort(softm, descending=True)
    srtd_group = torch.argsort(softm_group, descending=True)

    outs = []
    outs_group = []
    for i in range(top_n):
        prob = round(softm[srtd[i]].item(), 2)
        prob_group = round(softm_group[srtd_group[i]].item(), 2)
        outs.append((lang_dict[str(srtd[i].item())], prob))
        outs_group.append((lang_group_dict[str(srtd_group[i].item())], prob_group))

    return outs, outs_group


class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sample_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sample_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_silence_samples = sample_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sample_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """
        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sample_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sample_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sample_rate, 1)}

        return None


def collect_chunks(tss: List[dict],
                   wav: torch.Tensor):
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']: i['end']])
    return torch.cat(chunks)


def drop_chunks(tss: List[dict],
                wav: torch.Tensor):
    chunks = []
    cur_start = 0
    for i in tss:
        chunks.append((wav[cur_start: i['start']]))
        cur_start = i['end']
    return torch.cat(chunks)
