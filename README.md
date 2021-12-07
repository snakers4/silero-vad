[![Mailing list : test](http://img.shields.io/badge/Email-gray.svg?style=for-the-badge&logo=gmail)](mailto:hello@silero.ai) [![Mailing list : test](http://img.shields.io/badge/Telegram-blue.svg?style=for-the-badge&logo=telegram)](https://t.me/silero_speech) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-MIT-lightgrey.svg?style=for-the-badge)](https://github.com/snakers4/silero-vad/blob/master/LICENSE)

[![Open on Torch Hub](https://img.shields.io/badge/Torch-Hub-red?logo=pytorch&style=for-the-badge)](https://pytorch.org/hub/snakers4_silero-vad_vad/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb)

![header](https://user-images.githubusercontent.com/12515440/89997349-b3523080-dc94-11ea-9906-ca2e8bc50535.png)

<center> <h1>Silero VAD</h1> </center>

**Silero VAD** - pre-trained enterprise-grade [Voice Activity Detector](https://en.wikipedia.org/wiki/Voice_activity_detection) (also see our [STT models](https://github.com/snakers4/silero-models)).

https://user-images.githubusercontent.com/36505480/144874384-95f80f6d-a4f1-42cc-9be7-004c891dd481.mp4

<center> <h3>Key features</h3> </center>

- **High accuracy**

  Silero VAD shows an [excellent result](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics#vs-other-available-solutions)  for speech detection in streaming tasks.
- **Fast**

  One audio chunk (30+ ms) [takes](https://github.com/snakers4/silero-vad/wiki/Performance-Metrics#silero-vad-performance-metrics) **1ms** to be processed on a single CPU thread. Using batching and/or GPU one can greatly speed up inference time in production tasks.

- **Lightweight**

  JIT model size is less than one megabyte.

- **Generalized**

  Silero VAD was trained on a big corpora that included over **100** languages and performs well on audio of varying backgorund noise levels.

- **Variable sampling rate**

  Silero VAD [supports](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics#sample-rate-comparison)  **8000** and **16000** [sampling rate](https://en.wikipedia.org/wiki/Sampling_(signal_processing)#Sampling_rate)

- **Variable chunk size**

  Model was trained on audio chunks of variable lengths. Chunks of length **30 ms**, **60 ms** and **100 ms** are supported directly, other may perform well too.


<center> <h3>Typical use cases</h3> </center>
<br/><br/>

- Voice activity detection for IOT / edge / mobile use cases
- Data cleaning and preparation, voice detection in general

<br/><br/>
<center> <h3>Links</h3> </center>

- [Examples and Dependencies](https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies#dependencies)
- [Quality Metrics](https://github.com/snakers4/silero-vad/wiki/Quality-Metrics)
- [Performance Metrics](https://github.com/snakers4/silero-vad/wiki/Performance-Metrics)
- Number Detector and Language classifier [models](https://github.com/snakers4/silero-vad/wiki/Other-Models)
- [Versions and Available Models](https://github.com/snakers4/silero-vad/wiki/Version-history-and-Available-Models)


<center> <h3>Get in touch</h3> </center>

Try our models, create an [issue](https://github.com/snakers4/silero-vad/issues/new), start a [discussion](https://github.com/snakers4/silero-vad/discussions/new), join our telegram [chat](https://t.me/silero_speech), [email](mailto:hello@silero.ai) us, read our [news](https://t.me/silero_news).
Please see our [wiki](https://github.com/snakers4/silero-models/wiki) and [tiers](https://github.com/snakers4/silero-models/wiki/Licensing-and-Tiers) for relevant information and [email](mailto:hello@silero.ai) us directly.

Please see our [wiki](https://github.com/snakers4/silero-models/wiki) and [tiers](https://github.com/snakers4/silero-models/wiki/Licensing-and-Tiers) for relevant information and [email](mailto:hello@silero.ai) us directly.

**Citations**

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
