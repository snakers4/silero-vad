import torch
import torchaudio
from typing import List
from itertools import repeat
from collections import deque
import torch.nn.functional as F


torchaudio.set_audio_backend("soundfile")  # switch backend


languages = ['ru', 'en', 'de', 'es']


class IterativeMedianMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.median = 0
        self.counts = {}
        for i in range(0, 101, 1):
            self.counts[i / 100] = 0
        self.total_values = 0

    def __call__(self, val):
        self.total_values += 1
        rounded = round(abs(val), 2)
        self.counts[rounded] += 1
        bin_sum = 0
        for j in self.counts:
            bin_sum += self.counts[j]
            if bin_sum >= self.total_values / 2:
                self.median = j
                break
        return self.median


def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs


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


def save_audio(path: str,
               tensor: torch.Tensor,
               sr: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sr)


def init_jit_model(model_path: str,
                   device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def get_speech_ts(wav: torch.Tensor,
                  model,
                  trig_sum: float = 0.25,
                  neg_trig_sum: float = 0.07,
                  num_steps: int = 8,
                  batch_size: int = 200,
                  num_samples_per_window: int = 4000,
                  min_speech_samples: int = 10000, #samples
                  min_silence_samples: int = 500,
                  run_function=validate,
                  visualize_probs=False):

    num_samples = num_samples_per_window
    assert num_samples % num_steps == 0
    step = int(num_samples / num_steps)  # stride / hop
    outs = []
    to_concat = []
    for i in range(0, len(wav), step):
        chunk = wav[i: i+num_samples]
        if len(chunk) < num_samples:
            chunk = F.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk.unsqueeze(0))
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0))
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []

    if to_concat:
        chunks = torch.Tensor(torch.cat(to_concat, dim=0))
        out = run_function(model, chunks)
        outs.append(out)

    outs = torch.cat(outs, dim=0)

    buffer = deque(maxlen=num_steps)  # maxlen reached => first element dropped
    triggered = False
    speeches = []
    current_speech = {}
    if visualize_probs:
      import pandas as pd
      smoothed_probs = []

    speech_probs = outs[:, 1]  # this is very misleading
    temp_end = 0
    for i, predict in enumerate(speech_probs):  # add name
        buffer.append(predict)
        smoothed_prob = (sum(buffer) / len(buffer))
        if visualize_probs:
          smoothed_probs.append(float(smoothed_prob))
        if (smoothed_prob >= trig_sum) and temp_end:
            temp_end=0
        if (smoothed_prob >= trig_sum) and not triggered:
            triggered = True
            current_speech['start'] = step * max(0, i-num_steps)
            continue
        if (smoothed_prob < neg_trig_sum) and triggered:
            if not temp_end:
                temp_end = step * i
            if step * i - temp_end < min_silence_samples:
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
        current_speech['end'] = len(wav)
        speeches.append(current_speech)
    
    if visualize_probs:
      pd.DataFrame({'probs':smoothed_probs}).plot(figsize=(16,8))
    return speeches


def get_speech_ts_adaptive(wav: torch.Tensor,
                      model,
                      batch_size: int = 200,
                      step: int = 500,
                      num_samples_per_window: int = 4000, # Number of samples per audio chunk to feed to NN (4000 for 16k SR, 2000 for 8k SR is optimal)
                      min_speech_samples: int = 10000,  # samples
                      min_silence_samples: int = 4000,
                      speech_pad_samples: int = 2000,
                      run_function=validate,
                      visualize_probs=False,
                      device='cpu'):
    """
    This function is used for splitting long audios into speech chunks using silero VAD
    Attention! All default sample rate values are optimal for 16000 sample rate model, if you are using 8000 sample rate model optimal values are half as much!

    Parameters
    ----------
    batch_size: int
        batch size to feed to silero VAD (default - 200)

    step: int
        step size in samples, (default - 500)

    num_samples_per_window: int
        window size in samples (chunk length in samples to feed to NN, default - 4000)

    min_speech_samples: int
        if speech duration is shorter than this value, do not consider it speech (default - 10000)

    min_silence_samples: int
        number of samples to wait before considering as the end of speech (default - 4000)

    speech_pad_samples: int
        widen speech by this amount of samples each side (default - 2000)

    run_function: function
        function to use for the model call

    visualize_probs: bool
        whether draw prob hist or not (default: False)

    device: string
        torch device to use for the model call (default - "cpu")

    Returns
    ----------
    speeches: list
        list containing ends and beginnings of speech chunks (in samples)
    """
    if visualize_probs:
      import pandas as pd    

    num_samples = num_samples_per_window
    num_steps = int(num_samples / step)
    assert min_silence_samples >= step
    outs = []
    to_concat = []
    for i in range(0, len(wav), step):
        chunk = wav[i: i+num_samples]
        if len(chunk) < num_samples:
            chunk = F.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk.unsqueeze(0))
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []

    if to_concat:
        chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
        out = run_function(model, chunks)
        outs.append(out)

    outs = torch.cat(outs, dim=0).cpu()

    buffer = deque(maxlen=num_steps)
    triggered = False
    speeches = []
    smoothed_probs = []
    current_speech = {}
    speech_probs = outs[:, 1]  # 0 index for silence probs, 1 index for speech probs
    median_probs = speech_probs.median()

    trig_sum = 0.89 * median_probs + 0.08 # 0.08 when median is zero, 0.97 when median is 1

    temp_end = 0
    for i, predict in enumerate(speech_probs):
        buffer.append(predict)
        smoothed_prob = max(buffer)
        if visualize_probs:
            smoothed_probs.append(float(smoothed_prob))
        if (smoothed_prob >= trig_sum) and temp_end:
            temp_end = 0
        if (smoothed_prob >= trig_sum) and not triggered:
            triggered = True
            current_speech['start'] = step * max(0, i-num_steps)
            continue
        if (smoothed_prob < trig_sum) and triggered:
            if not temp_end:
                temp_end = step * i
            if step * i - temp_end < min_silence_samples:
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
        current_speech['end'] = len(wav)
        speeches.append(current_speech)
    if visualize_probs:
        pd.DataFrame({'probs': smoothed_probs}).plot(figsize=(16, 8))

    for i, ts in enumerate(speeches):
        if i == 0:
            ts['start'] = max(0, ts['start'] - speech_pad_samples)
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - ts['end']
            if silence_duration < 2 * speech_pad_samples:
                ts['end'] += silence_duration // 2
                speeches[i+1]['start'] = max(0, speeches[i+1]['start'] - silence_duration // 2)
            else:
                ts['end'] += speech_pad_samples
        else:
            ts['end'] = min(len(wav), ts['end'] + speech_pad_samples)

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


class VADiterator:
    def __init__(self,
                 trig_sum: float = 0.26,
                 neg_trig_sum: float = 0.07,
                 num_steps: int = 8,
                 num_samples_per_window: int = 4000):
        self.num_samples = num_samples_per_window
        self.num_steps = num_steps
        assert self.num_samples % num_steps == 0
        self.step = int(self.num_samples / num_steps)   # 500 samples is good enough
        self.prev = torch.zeros(self.num_samples)
        self.last = False
        self.triggered = False
        self.buffer = deque(maxlen=num_steps)
        self.num_frames = 0
        self.trig_sum = trig_sum
        self.neg_trig_sum = neg_trig_sum
        self.current_name = ''

    def refresh(self):
        self.prev = torch.zeros(self.num_samples)
        self.last = False
        self.triggered = False
        self.buffer = deque(maxlen=self.num_steps)
        self.num_frames = 0

    def prepare_batch(self, wav_chunk, name=None):
        if (name is not None) and (name != self.current_name):
            self.refresh()
            self.current_name = name
        assert len(wav_chunk) <= self.num_samples
        self.num_frames += len(wav_chunk)
        if len(wav_chunk) < self.num_samples:
            wav_chunk = F.pad(wav_chunk, (0, self.num_samples - len(wav_chunk)))  # short chunk => eof audio
            self.last = True

        stacked = torch.cat([self.prev, wav_chunk])
        self.prev = wav_chunk

        overlap_chunks = [stacked[i:i+self.num_samples].unsqueeze(0)
                          for i in range(self.step, self.num_samples+1, self.step)]
        return torch.cat(overlap_chunks, dim=0)

    def state(self, model_out):
        current_speech = {}
        speech_probs = model_out[:, 1]  # this is very misleading
        for i, predict in enumerate(speech_probs):
            self.buffer.append(predict)
            if ((sum(self.buffer) / len(self.buffer)) >= self.trig_sum) and not self.triggered:
                self.triggered = True
                current_speech[self.num_frames - (self.num_steps-i) * self.step] = 'start'
            if ((sum(self.buffer) / len(self.buffer)) < self.neg_trig_sum) and self.triggered:
                current_speech[self.num_frames - (self.num_steps-i) * self.step] = 'end'
                self.triggered = False
        if self.triggered and self.last:
            current_speech[self.num_frames] = 'end'
        if self.last:
            self.refresh()
        return current_speech, self.current_name


class VADiteratorAdaptive:
    def __init__(self,
                 trig_sum: float = 0.26,
                 neg_trig_sum: float = 0.06,
                 step: int = 500,
                 num_samples_per_window: int = 4000,
                 speech_pad_samples: int = 1000,
                 accum_period: int = 50):
        """
        This class is used for streaming silero VAD usage

        Parameters
        ----------
        trig_sum: float
            trigger value for speech probability, probs above this value are considered speech, switch to TRIGGERED state (default - 0.26)

        neg_trig_sum: float
            in triggered state probabilites below this value are considered nonspeech, switch to NONTRIGGERED state (default - 0.06)

        step: int
            step size in samples, (default - 500)

        num_samples_per_window: int
            window size in samples (chunk length in samples to feed to NN, default - 4000)

        speech_pad_samples: int
            widen speech by this amount of samples each side (default - 1000)

        accum_period: int
            number of chunks / iterations to wait before switching from constant (initial) trig and neg_trig coeffs to adaptive median coeffs (default - 50) 

        """
        self.num_samples = num_samples_per_window
        self.num_steps = int(num_samples_per_window / step)
        self.step = step
        self.prev = torch.zeros(self.num_samples)
        self.last = False
        self.triggered = False
        self.buffer = deque(maxlen=self.num_steps)
        self.num_frames = 0
        self.trig_sum = trig_sum
        self.neg_trig_sum = neg_trig_sum
        self.current_name = ''
        self.median_meter = IterativeMedianMeter()
        self.median = 0
        self.total_steps = 0
        self.accum_period = accum_period
        self.speech_pad_samples = speech_pad_samples

    def refresh(self):
        self.prev = torch.zeros(self.num_samples)
        self.last = False
        self.triggered = False
        self.buffer = deque(maxlen=self.num_steps)
        self.num_frames = 0
        self.median_meter.reset()
        self.median = 0
        self.total_steps = 0

    def prepare_batch(self, wav_chunk, name=None):
        if (name is not None) and (name != self.current_name):
            self.refresh()
            self.current_name = name
        assert len(wav_chunk) <= self.num_samples
        self.num_frames += len(wav_chunk)
        if len(wav_chunk) < self.num_samples:
            wav_chunk = F.pad(wav_chunk, (0, self.num_samples - len(wav_chunk)))  # short chunk => eof audio
            self.last = True

        stacked = torch.cat([self.prev, wav_chunk])
        self.prev = wav_chunk

        overlap_chunks = [stacked[i:i+self.num_samples].unsqueeze(0)
                          for i in range(self.step, self.num_samples+1, self.step)]
        return torch.cat(overlap_chunks, dim=0)

    def state(self, model_out):
        current_speech = {}
        speech_probs = model_out[:, 1]  # 0 index for silence probs, 1 index for speech probs
        for i, predict in enumerate(speech_probs):
            self.median = self.median_meter(predict.item())
            if self.total_steps < self.accum_period:
                trig_sum = self.trig_sum
                neg_trig_sum = self.neg_trig_sum
            else:
                trig_sum = 0.89 * self.median + 0.08 # 0.08 when median is zero, 0.97 when median is 1
                neg_trig_sum = 0.6 * self.median
            self.total_steps += 1
            self.buffer.append(predict)
            smoothed_prob = max(self.buffer)
            if (smoothed_prob >= trig_sum) and not self.triggered:
                self.triggered = True
                current_speech[max(0, self.num_frames - (self.num_steps-i) * self.step - self.speech_pad_samples)] = 'start'
            if (smoothed_prob < neg_trig_sum) and self.triggered:
                current_speech[self.num_frames - (self.num_steps-i) * self.step + self.speech_pad_samples] = 'end'
                self.triggered = False
        if self.triggered and self.last:
            current_speech[self.num_frames] = 'end'
        if self.last:
            self.refresh()
        return current_speech, self.current_name


def state_generator(model,
                    audios: List[str],
                    onnx: bool = False,
                    trig_sum: float = 0.26,
                    neg_trig_sum: float = 0.07,
                    num_steps: int = 8,
                    num_samples_per_window: int = 4000,
                    audios_in_stream: int = 2,
                    run_function=validate):
    VADiters = [VADiterator(trig_sum, neg_trig_sum, num_steps, num_samples_per_window) for i in range(audios_in_stream)]
    for i, current_pieces in enumerate(stream_imitator(audios, audios_in_stream, num_samples_per_window)):
        for_batch = [x.prepare_batch(*y) for x, y in zip(VADiters, current_pieces)]
        batch = torch.cat(for_batch)

        outs = run_function(model, batch)
        vad_outs = torch.split(outs, num_steps)

        states = []
        for x, y in zip(VADiters, vad_outs):
            cur_st = x.state(y)
            if cur_st[0]:
                states.append(cur_st)
        yield states


def stream_imitator(audios: List[str],
                    audios_in_stream: int,
                    num_samples_per_window: int = 4000):
    audio_iter = iter(audios)
    iterators = []
    num_samples = num_samples_per_window
    # initial wavs
    for i in range(audios_in_stream):
        next_wav = next(audio_iter)
        wav = read_audio(next_wav)
        wav_chunks = iter([(wav[i:i+num_samples], next_wav) for i in range(0, len(wav), num_samples)])
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
                    next_wav = next(audio_iter)
                    print('Loading next wav: ', next_wav)
                    wav = read_audio(next_wav)
                    iterators[i] = iter([(wav[i:i+num_samples], next_wav) for i in range(0, len(wav), num_samples)])
                    out, wav_name = next(iterators[i])
                except StopIteration:
                    good_iters -= 1
                    iterators[i] = repeat((torch.zeros(num_samples), 'junk'))
                    out, wav_name = next(iterators[i])
                    if good_iters == 0:
                        return
            values.append((out, wav_name))
        yield values


def single_audio_stream(model,
                        audio: torch.Tensor,
                        num_samples_per_window:int = 4000,
                        run_function=validate,
                        iterator_type='basic',
                        **kwargs):
    
    num_samples = num_samples_per_window
    if iterator_type == 'basic':
        VADiter = VADiterator(num_samples_per_window=num_samples_per_window, **kwargs)
    elif iterator_type == 'adaptive':
        VADiter = VADiteratorAdaptive(num_samples_per_window=num_samples_per_window, **kwargs)
        
    wav = read_audio(audio)
    wav_chunks = iter([wav[i:i+num_samples] for i in range(0, len(wav), num_samples)])
    for chunk in wav_chunks:
        batch = VADiter.prepare_batch(chunk)

        outs = run_function(model, batch)

        states = []
        state = VADiter.state(outs)
        if state[0]:
            states.append(state[0])
        yield states


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
