[![Mailing list : test](http://img.shields.io/badge/Email-gray.svg?style=for-the-badge&logo=gmail)](mailto:hello@silero.ai) [![Mailing list : test](http://img.shields.io/badge/Telegram-blue.svg?style=for-the-badge&logo=telegram)](https://t.me/silero_speech) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-MIT-lightgrey.svg?style=for-the-badge)](https://github.com/snakers4/silero-vad/blob/master/LICENSE) 
 
[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-vad_vad/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb)

![header](https://user-images.githubusercontent.com/12515440/89997349-b3523080-dc94-11ea-9906-ca2e8bc50535.png)

- [Silero VAD](#silero-vad)
  - [TLDR](#tldr)
    - [Live Demonstration](#live-demonstration)
  - [Getting Started](#getting-started)
    - [Pre-trained Models](#pre-trained-models)
    - [Version History](#version-history)
    - [PyTorch](#pytorch)
      - [VAD](#vad)
      - [Number Detector](#number-detector)
      - [Language Classifier](#language-classifier)
    - [ONNX](#onnx)
      - [VAD](#vad-1)
      - [Number Detector](#number-detector-1)
      - [Language Classifier](#language-classifier-1)
  - [Metrics](#metrics)
    - [Performance Metrics](#performance-metrics)
      - [Streaming Latency](#streaming-latency)
      - [Full Audio Throughput](#full-audio-throughput)
    - [VAD Quality Metrics](#vad-quality-metrics)
  - [FAQ](#faq)
    - [VAD Parameter Fine Tuning](#vad-parameter-fine-tuning)
      - [Classic way](#classic-way)
      - [Adaptive way](#adaptive-way)
    - [How VAD Works](#how-vad-works)
    - [VAD Quality Metrics Methodology](#vad-quality-metrics-methodology)
    - [How Number Detector Works](#how-number-detector-works)
    - [How Language Classifier Works](#how-language-classifier-works)
  - [Contact](#contact)
    - [Get in Touch](#get-in-touch)
    - [Commercial Inquiries](#commercial-inquiries)
  - [References](#references)
  - [Citations](#citations)


# Silero VAD
![image](https://user-images.githubusercontent.com/36505480/107667211-06cf2680-6c98-11eb-9ee5-37eb4596260f.png)

## TLDR

**Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier.**
Enterprise-grade Speech Products made refreshingly simple (also see our [STT](https://github.com/snakers4/silero-models) models).

Currently, there are hardly any high quality / modern / free / public voice activity detectors except for WebRTC Voice Activity Detector ([link](https://github.com/wiseman/py-webrtcvad)). WebRTC though starts to show its age and it suffers from many false positives.

Also in some cases it is crucial to be able to anonymize large-scale spoken corpora (i.e. remove personal data). Typically personal data is considered to be private / sensitive if it contains (i) a name (ii) some private ID. Name recognition is a highly subjective matter and it depends on locale and business case, but Voice Activity and Number Detection are quite general tasks.

**Key features:**

- Modern, portable;
- Low memory footprint;
- Superior metrics to WebRTC;
- Trained on huge spoken corpora and noise / sound libraries;
- Slower than WebRTC, but fast enough for IOT / edge / mobile applications;
- Unlike WebRTC (which mostly tells silence from voice), our VAD can tell voice from noise / music / silence;

**Typical use cases:**

- Spoken corpora anonymization;
- Can be used together with WebRTC;
- Voice activity detection for IOT / edge / mobile use cases;
- Data cleaning and preparation, number and voice detection in general;
- PyTorch and ONNX can be used with a wide variety of deployment options and backends in mind; 

### Live Demonstration

For more information, please see [examples](https://github.com/snakers4/silero-vad/tree/master/examples).

https://user-images.githubusercontent.com/28188499/116685087-182ff100-a9b2-11eb-927d-ed9f621226ee.mp4

https://user-images.githubusercontent.com/8079748/117580455-4622dd00-b0f8-11eb-858d-e6368ed4eada.mp4

## Getting Started

The models are small enough to be included directly into this repository. Newer models will supersede older models directly.

### Pre-trained Models

**Currently we provide the following endpoints:**

| model=                     | Params | Model type          | Streaming | Languages                  | PyTorch            | ONNX               | Colab                                                                                                                                                                   |
| -------------------------- | ------ | ------------------- | --------- | -------------------------- | ------------------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `'silero_vad'`             | 1.1M   | VAD                 | Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_micro'`       | 10K    | VAD                 | Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_micro_8k'`    | 10K    | VAD                 | Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_mini'`        | 100K   | VAD                 | Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_mini_8k'`     | 100K   | VAD                 | Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_number_detector'` | 1.1M   | Number Detector     | No        | `ru`, `en`, `de`, `es`     | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_lang_detector'`   | 1.1M   | Language Classifier | No        | `ru`, `en`, `de`, `es`     | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| ~~`'silero_lang_detector_116'`~~   | ~~1.7M~~   | ~~Language Classifier~~ ||| | ||
| `'silero_lang_detector_95'`   | 4.7M   | Language Classifier | No        |   [95 languages](https://github.com/snakers4/silero-vad/blob/master/files/lang_dict_95.json)   | :heavy_check_mark: | :heavy_check_mark: | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |

(*) Though explicitly trained on these languages, VAD should work on any Germanic, Romance or Slavic Languages out of the box.

What models do:

- VAD - detects speech;
- Number Detector  - detects spoken numbers (i.e. thirty five);
- Language Classifier - classifies utterances between language;
- Language Classifier 95 - classifies among 95 languages as well as 58 language groups (mutually intelligible languages -> same group)

### Version History

**Version history:**

| Version | Date       | Comment                                                                                                                     |
| ------- | ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| `v1`    | 2020-12-15 | Initial release                                                                                                             |
| `v1.1`  | 2020-12-24 | better vad models compatible with chunks shorter than 250 ms                                                                |
| `v1.2`  | 2020-12-30 | Number Detector added                                                                                                       |
| `v2`    | 2021-01-11 | Add Language Classifier heads (en, ru, de, es)                                                                              |
| `v2.1`  | 2021-02-11 | Add micro (10k params) VAD models                                                                                           |
| `v2.2`  | 2021-03-22 | Add micro 8000 sample rate VAD models                                                                                       |
| `v2.3`  | 2021-04-12 | Add mini (100k params) VAD models (8k and 16k sample rate)  + **new** adaptive utils for full audio and single audio stream |
| `v2.4`  | 2021-07-09 | Add 116 languages classifier and group classifier |
| `v2.4`  | 2021-07-09 | Deleted 116 language classifier, added 95 language classifier instead (get rid of lowspoken languages for quality improvement) 
|

### PyTorch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb)

We are keeping the colab examples up-to-date, but you can manually manage your dependencies:

- `pytorch` >= 1.7.1 (there were breaking changes in `torch.hub` introduced in 1.7);
- `torchaudio` >= 0.7.2 (used only for IO and resampling, can be easily replaced);
- `soundfile` >= 0.10.3 (used as a default backend for torchaudio, can be replaced);

All of the dependencies except for PyTorch are superficial and for utils / example only. You can use any libraries / pipelines that read files and resample into 16 kHz.

#### VAD

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-vad_vad/)

```python
import torch
torch.set_num_threads(1)
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_ts,
 get_speech_ts_adaptive,
 _, read_audio,
 _, _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/en.wav')
# full audio
# get speech timestamps from full audio file

# classic way
speech_timestamps = get_speech_ts(wav, model,
                                  num_steps=4)
pprint(speech_timestamps)

# adaptive way
speech_timestamps = get_speech_ts_adaptive(wav, model)
pprint(speech_timestamps)
```

#### Number Detector

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-vad_number/)

```python
import torch
torch.set_num_threads(1)
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_number_detector',
                              force_reload=True)

(get_number_ts,
 _, read_audio,
 _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/en_num.wav')
# full audio
# get number timestamps from full audio file
number_timestamps = get_number_ts(wav, model)

pprint(number_timestamps)
```

#### Language Classifier 
##### 4 languages

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-vad_language/)

```python
import torch
torch.set_num_threads(1)
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_lang_detector',
                              force_reload=True)

get_language, read_audio = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/de.wav')
language = get_language(wav, model)

pprint(language)
```

##### 95 languages

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-vad_language/)

```python
import torch
torch.set_num_threads(1)
from pprint import pprint

model, lang_dict, lang_group_dict,  utils = torch.hub.load(
                              repo_or_dir='snakers4/silero-vad',
                              model='silero_lang_detector_95',
                              force_reload=True)

get_language_and_group, read_audio = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/de.wav')
languages, language_groups = get_language_and_group(wav, model, lang_dict, lang_group_dict, top_n=2)

for i in languages:
  pprint(f'Language: {i[0]} with prob {i[-1]}')

for i in language_groups:
  pprint(f'Language group: {i[0]} with prob {i[-1]}')
```

### ONNX

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb)

You can run our models everywhere, where you can import the ONNX model or run ONNX runtime.

#### VAD

```python
import torch
import onnxruntime
from pprint import pprint

_, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

(get_speech_ts,
 get_speech_ts_adaptive,
 _, read_audio,
 _, _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)

def validate_onnx(model, inputs):
    with torch.no_grad():
        ort_inputs = {'input': inputs.cpu().numpy()}
        outs = model.run(None, ort_inputs)
        outs = [torch.Tensor(x) for x in outs]
    return outs[0]
    
model = init_onnx_model(f'{files_dir}/model.onnx')
wav = read_audio(f'{files_dir}/en.wav')

# get speech timestamps from full audio file

# classic way
speech_timestamps = get_speech_ts(wav, model, num_steps=4, run_function=validate_onnx) 
pprint(speech_timestamps)

# adaptive way
speech_timestamps = get_speech_ts(wav, model, run_function=validate_onnx) 
pprint(speech_timestamps)
```

#### Number Detector

```python
import torch
import onnxruntime
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_number_detector',
                              force_reload=True)

(get_number_ts,
 _, read_audio,
 _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)

def validate_onnx(model, inputs):
    with torch.no_grad():
        ort_inputs = {'input': inputs.cpu().numpy()}
        outs = model.run(None, ort_inputs)
        outs = [torch.Tensor(x) for x in outs]
    return outs
    
model = init_onnx_model(f'{files_dir}/number_detector.onnx')
wav = read_audio(f'{files_dir}/en_num.wav')

# get speech timestamps from full audio file
number_timestamps = get_number_ts(wav, model, run_function=validate_onnx) 
pprint(number_timestamps)
```

#### Language Classifier
##### 4 languages

```python
import torch
import onnxruntime
from pprint import pprint

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_lang_detector',
                              force_reload=True)
                              
get_language, read_audio = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)

def validate_onnx(model, inputs):
    with torch.no_grad():
        ort_inputs = {'input': inputs.cpu().numpy()}
        outs = model.run(None, ort_inputs)
        outs = [torch.Tensor(x) for x in outs]
    return outs
    
model = init_onnx_model(f'{files_dir}/number_detector.onnx')
wav = read_audio(f'{files_dir}/de.wav')

language = get_language(wav, model, run_function=validate_onnx)
print(language)
```

##### 95 languages

```python
import torch
import onnxruntime
from pprint import pprint

model, lang_dict, lang_group_dict,  utils = torch.hub.load(
                              repo_or_dir='snakers4/silero-vad',
                              model='silero_lang_detector_95',
                              force_reload=True)
                              
get_language_and_group, read_audio = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

def init_onnx_model(model_path: str):
    return onnxruntime.InferenceSession(model_path)

def validate_onnx(model, inputs):
    with torch.no_grad():
        ort_inputs = {'input': inputs.cpu().numpy()}
        outs = model.run(None, ort_inputs)
        outs = [torch.Tensor(x) for x in outs]
    return outs
    
model = init_onnx_model(f'{files_dir}/lang_classifier_95.onnx')
wav = read_audio(f'{files_dir}/de.wav')

languages, language_groups = get_language_and_group(wav, model, lang_dict, lang_group_dict, top_n=2, run_function=validate_onnx)

for i in languages:
  pprint(f'Language: {i[0]} with prob {i[-1]}')

for i in language_groups:
  pprint(f'Language group: {i[0]} with prob {i[-1]}')

```
[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-vad_language/)

## Metrics

### Performance Metrics

All speed test were run on AMD Ryzen Threadripper 3960X using only 1 thread: 
```
torch.set_num_threads(1) # pytorch
ort_session.intra_op_num_threads = 1 # onnx
ort_session.inter_op_num_threads = 1 # onnx
```

#### Streaming Latency

Streaming latency depends on 2 variables:

- **num_steps** - number of windows to split each audio chunk into. Our post-processing class keeps previous chunk in memory (250 ms), so new chunk (also 250 ms) is appended to it. The resulting big chunk (500 ms) is split into **num_steps** overlapping windows, each 250 ms long.

- **number of audio streams**

So **batch size** for streaming is **num_steps * number of audio streams**. Time between receiving new audio chunks and getting results is shown in picture:

| Batch size | Pytorch model time, ms | Onnx model time, ms |
| :--------: | :--------------------: | :-----------------: |
|   **2**    |           9            |          2          |
|   **4**    |           11           |          4          |
|   **8**    |           14           |          7          |
|   **16**   |           19           |         12          |
|   **40**   |           36           |         29          |
|   **80**   |           64           |         55          |
|  **120**   |           96           |         85          |
|  **200**   |          157           |         137         |

#### Full Audio Throughput

**RTS** (seconds of audio processed per second, real time speed, or 1 / RTF) for full audio processing depends on **num_steps** (see previous paragraph) and **batch size** (bigger is better).

| Batch size | num_steps | Pytorch model RTS | Onnx model RTS |
| :--------: | :-------: | :---------------: | :------------: |
|   **40**   |   **4**   |        68         |       86       |
|   **40**   |   **8**   |        34         |       43       |
|   **80**   |   **4**   |        78         |       91       |
|   **80**   |   **8**   |        39         |       45       |
|  **120**   |   **4**   |        78         |       88       |
|  **120**   |   **8**   |        39         |       44       |
|  **200**   |   **4**   |        80         |       91       |
|  **200**   |   **8**   |        40         |       46       |

### VAD Quality Metrics

We use random 250 ms audio chunks for validation. Speech to non-speech ratio among chunks is about ~50/50 (i.e. balanced). Speech chunks are sampled from real audios in four different languages (English, Russian, Spanish, German), then random background noise is added to some of them (~40%). 

Since our VAD (only VAD, other networks are more flexible) was trained on chunks of the same length, model's output is just one float from 0 to 1 - **speech probability**. We use speech probabilities as thresholds for precision-recall curve. This can be extended to 100 - 150 ms. Less than 100 - 150 ms cannot be distinguished as speech with confidence.

[Webrtc](https://github.com/wiseman/py-webrtcvad) splits audio into frames, each frame has corresponding number (0 **or** 1). We use 30ms frames for webrtc, so each 250 ms chunk is split into 8 frames, their **mean** value is used as a threshold for plot.

[Auditok](https://github.com/amsehili/auditok) - logic same as Webrtc, but we use 50ms frames.

![image](https://user-images.githubusercontent.com/36505480/107667211-06cf2680-6c98-11eb-9ee5-37eb4596260f.png)

## FAQ

### VAD Parameter Fine Tuning

#### Classic way

**This is straightforward classic method `get_speech_ts` where thresholds (`trig_sum` and `neg_trig_sum`) are specified by users**
- Among others, we provide several [utils](https://github.com/snakers4/silero-vad/blob/8b28767292b424e3e505c55f15cd3c4b91e4804b/utils.py#L52-L59) to simplify working with VAD;
- We provide sensible basic hyper-parameters that work for us, but your case can be different;
- `trig_sum` - overlapping windows are used for each audio chunk, trig sum defines average probability among those windows for switching into triggered state (speech state);
- `neg_trig_sum` - same as `trig_sum`, but for switching from triggered to non-triggered state (non-speech)
- `num_steps` - nubmer of overlapping windows to split audio chunk into (we recommend 4 or 8)
- `num_samples_per_window` - number of samples in each window, our models were trained using `4000` samples (250 ms) per window, so this is preferable value (lesser values reduce [quality](https://github.com/snakers4/silero-vad/issues/2#issuecomment-750840434));
- `min_speech_samples` - minimum speech chunk duration in samples
- `min_silence_samples` - minimum silence duration in samples between to separate speech chunks

Optimal parameters may vary per domain, but we provided a tiny tool to learn the best parameters. You can invoke `speech_timestamps` with visualize_probs=True (`pandas` required):

```
speech_timestamps = get_speech_ts(wav, model,
                                  num_samples_per_window=4000,
                                  num_steps=4,
                                  visualize_probs=True)
```

#### Adaptive way

**Adaptive algorithm (`get_speech_ts_adaptive`) automatically selects thresholds (`trig_sum` and `neg_trig_sum`) based on median speech probabilities over the whole audio, SOME ARGUMENTS VARY FROM THE CLASSIC WAY FUNCTION ARGUMENTS**
- `batch_size` - batch size to feed to silero VAD (default - `200`)
- `step` - step size in samples, (default - `500`) (`num_samples_per_window` / `num_steps` from classic method)
- `num_samples_per_window` -  number of samples in each window, our models were trained using `4000` samples (250 ms) per window, so this is preferable value (lesser values reduce [quality](https://github.com/snakers4/silero-vad/issues/2#issuecomment-750840434));
- `min_speech_samples` - minimum speech chunk duration in samples (default - `10000`)
- `min_silence_samples` - minimum silence duration in samples between to separate speech chunks (default - `4000`)
- `speech_pad_samples` - widen speech by this amount of samples each side (default - `2000`)

```
speech_timestamps = get_speech_ts_adaptive(wav, model,
                                  num_samples_per_window=4000,
                                  step=500,
                                  visualize_probs=True)
```


The chart should looks something like this:

![image](https://user-images.githubusercontent.com/12515440/106242896-79142580-6219-11eb-9add-fa7195d6fd26.png)

With this particular example you can try shorter chunks (`num_samples_per_window=1600`), but this results in too much noise:

![image](https://user-images.githubusercontent.com/12515440/106243014-a8c32d80-6219-11eb-8374-969f372807f1.png)


### How VAD Works

- Audio is split into 250 ms chunks (you can choose any chunk size, but quality with chunks shorter than 100ms will suffer and there will be more false positives and "unnatural" pauses);
- VAD keeps record of a previous chunk (or zeros at the beginning of the stream);
- Then this 500 ms audio (250 ms + 250 ms) is split into N (typically 4 or 8) windows and the model is applied to this window batch. Each window is 250 ms long (naturally, windows overlap);
- Then probability is averaged across these windows;
- Though typically pauses in speech are 300 ms+ or longer (pauses less than 200-300ms are typically not meaninful), it is hard to confidently classify speech vs noise / music on very short chunks (i.e. 30 - 50ms);
- ~~We are working on lifting this limitation, so that you can use 100 - 125ms windows~~;

### VAD Quality Metrics Methodology

Please see [Quality Metrics](#quality-metrics)

### How Number Detector Works

- It is recommended to split long audio into short ones (< 15s) and apply model on each of them;
- Number Detector can classify if the whole audio contains a number, or if each audio frame contains a number;
- Audio is splitted into frames in a certain way, so, having a per-frame output, we can restore timing bounds for a numbers with an accuracy of about 0.2s;

### How Language Classifier Works

- **99%** validation accuracy
- Language classifier was trained using audio samples in 4 languages: **Russian**, **English**, **Spanish**, **German**
- More languages TBD
- Arbitrary audio length can be used, although network was trained using audio shorter than 15 seconds

### How Language Classifier 95 Works

- **85%** validation accuracy among 95 languages, **90%** validation accuracy among [58 language groups](https://github.com/snakers4/silero-vad/blob/master/files/lang_group_dict_95.json)
- Language classifier 95 was trained using audio samples in [95 languages](https://github.com/snakers4/silero-vad/blob/master/files/lang_dict_95.json)
- Arbitrary audio length can be used, although network was trained using audio shorter than 20 seconds

## Contact

### Get in Touch

Try our models, create an [issue](https://github.com/snakers4/silero-vad/issues/new), start a [discussion](https://github.com/snakers4/silero-vad/discussions/new), join our telegram [chat](https://t.me/silero_speech), [email](mailto:hello@silero.ai) us, read our [news](https://t.me/silero_news).

### Commercial Inquiries

Please see our [wiki](https://github.com/snakers4/silero-models/wiki) and [tiers](https://github.com/snakers4/silero-models/wiki/Licensing-and-Tiers) for relevant information and [email](mailto:hello@silero.ai) us directly.


## References

- Russian article - https://habr.com/ru/post/537274/
- English article - https://habr.com/ru/post/537276/
- Nice [thread](https://github.com/snakers4/silero-vad/discussions/16#discussioncomment-305830) in discussions

## Citations

```
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```
