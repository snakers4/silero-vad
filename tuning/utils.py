import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import random
import gc
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def read_audio(path: str,
               sampling_rate: int = 16000,
               normalize=False):

    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sampling_rate:
        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                       new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

    if normalize and wav.abs().max() != 0:
        wav = wav / wav.abs().max()

    return wav.squeeze(0)


def build_audiomentations_augs(p):
    from audiomentations import SomeOf, AirAbsorption, BandPassFilter, BandStopFilter, ClippingDistortion, HighPassFilter, HighShelfFilter, \
                                LowPassFilter, LowShelfFilter, Mp3Compression, PeakingFilter, PitchShift, RoomSimulator, SevenBandParametricEQ, \
                                Aliasing, AddGaussianNoise
    transforms = [Aliasing(p=1),
                  AddGaussianNoise(p=1),
                  AirAbsorption(p=1),
                  BandPassFilter(p=1),
                  BandStopFilter(p=1),
                  ClippingDistortion(p=1),
                  HighPassFilter(p=1),
                  HighShelfFilter(p=1),
                  LowPassFilter(p=1),
                  LowShelfFilter(p=1),
                  Mp3Compression(p=1),
                  PeakingFilter(p=1),
                  PitchShift(p=1),
                  RoomSimulator(p=1, leave_length_unchanged=True),
                  SevenBandParametricEQ(p=1)]
    tr = SomeOf((1, 3), transforms=transforms, p=p)
    return tr


class SileroVadDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train'):

        self.num_samples = 512  # constant, do not change
        self.sr = 16000  # constant, do not change

        self.resample_to_8k = config.tune_8k
        self.noise_loss = config.noise_loss
        self.max_train_length_sec = config.max_train_length_sec
        self.max_train_length_samples = config.max_train_length_sec * self.sr

        assert self.max_train_length_samples % self.num_samples == 0
        assert mode in ['train', 'val']

        dataset_path = config.train_dataset_path if mode == 'train' else config.val_dataset_path
        self.dataframe = pd.read_feather(dataset_path).reset_index(drop=True)
        self.index_dict = self.dataframe.to_dict('index')
        self.mode = mode
        print(f'DATASET SIZE : {len(self.dataframe)}')

        if mode == 'train':
            self.augs = build_audiomentations_augs(p=config.aug_prob)
        else:
            self.augs = None

    def __getitem__(self, idx):
        idx = None if self.mode == 'train' else idx
        wav, gt, mask = self.load_speech_sample(idx)

        if self.mode == 'train':
            wav = self.add_augs(wav)
            if len(wav) > self.max_train_length_samples:
                wav = wav[:self.max_train_length_samples]
                gt = gt[:int(self.max_train_length_samples / self.num_samples)]
                mask = mask[:int(self.max_train_length_samples / self.num_samples)]

        wav = torch.FloatTensor(wav)
        if self.resample_to_8k:
            transform = torchaudio.transforms.Resample(orig_freq=self.sr,
                                                       new_freq=8000)
            wav = transform(wav)
        return wav, torch.FloatTensor(gt), torch.from_numpy(mask)

    def __len__(self):
        return len(self.index_dict)

    def load_speech_sample(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self.index_dict) - 1)
        wav = read_audio(self.index_dict[idx]['audio_path'], self.sr).numpy()

        if len(wav) % self.num_samples != 0:
            pad_num = self.num_samples - (len(wav) % (self.num_samples))
            wav = np.pad(wav, (0, pad_num), 'constant', constant_values=0)

        gt, mask = self.get_ground_truth_annotated(self.index_dict[idx]['speech_ts'], len(wav))

        assert len(gt) == len(wav) / self.num_samples

        mask[gt == 0]

        return wav, gt, mask

    def get_ground_truth_annotated(self, annotation, audio_length_samples):
        gt = np.zeros(audio_length_samples)

        for i in annotation:
            gt[int(i['start'] * self.sr): int(i['end'] * self.sr)] = 1

        squeezed_predicts = np.average(gt.reshape(-1, self.num_samples), axis=1)
        squeezed_predicts = (squeezed_predicts > 0.5).astype(int)
        mask = np.ones(len(squeezed_predicts))
        mask[squeezed_predicts == 0] = self.noise_loss
        return squeezed_predicts, mask

    def add_augs(self, wav):
        while True:
            try:
                wav_aug = self.augs(wav, self.sr)
                if np.isnan(wav_aug.max()) or np.isnan(wav_aug.min()):
                    return wav
                return wav_aug
            except Exception as e:
                continue


def SileroVadPadder(batch):
    wavs = [batch[i][0] for i in range(len(batch))]
    labels = [batch[i][1] for i in range(len(batch))]
    masks = [batch[i][2] for i in range(len(batch))]

    wavs = torch.nn.utils.rnn.pad_sequence(
        wavs, batch_first=True, padding_value=0)

    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0)

    masks = torch.nn.utils.rnn.pad_sequence(
        masks, batch_first=True, padding_value=0)

    return wavs, labels, masks


class VADDecoderRNNJIT(nn.Module):

    def __init__(self):
        super(VADDecoderRNNJIT, self).__init__()

        self.rnn = nn.LSTMCell(128, 128)
        self.decoder = nn.Sequential(nn.Dropout(0.1),
                                     nn.ReLU(),
                                     nn.Conv1d(128, 1, kernel_size=1),
                                     nn.Sigmoid())

    def forward(self, x, state=torch.zeros(0)):
        x = x.squeeze(-1)
        if len(state):
            h, c = self.rnn(x, (state[0], state[1]))
        else:
            h, c = self.rnn(x)

        x = h.unsqueeze(-1).float()
        state = torch.stack([h, c])
        x = self.decoder(x)
        return x, state


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config,
          loader,
          jit_model,
          decoder,
          criterion,
          optimizer,
          device):

    losses = AverageMeter()
    decoder.train()

    context_size = 32 if config.tune_8k else 64
    num_samples = 256 if config.tune_8k else 512
    stft_layer = jit_model._model_8k.stft if config.tune_8k else jit_model._model.stft
    encoder_layer = jit_model._model_8k.encoder if config.tune_8k else jit_model._model.encoder

    with torch.enable_grad():
        for _, (x, targets, masks) in tqdm(enumerate(loader), total=len(loader)):
            targets = targets.to(device)
            x = x.to(device)
            masks = masks.to(device)
            x = torch.nn.functional.pad(x, (context_size, 0))

            outs = []
            state = torch.zeros(0)
            for i in range(context_size, x.shape[1], num_samples):
                input_ = x[:, i-context_size:i+num_samples]
                out = stft_layer(input_)
                out = encoder_layer(out)
                out, state = decoder(out, state)
                outs.append(out)
            stacked = torch.cat(outs, dim=2).squeeze(1)

            loss = criterion(stacked, targets)
            loss = (loss * masks).mean()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), masks.numel())

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg


def validate(config,
             loader,
             jit_model,
             decoder,
             criterion,
             device):

    losses = AverageMeter()
    decoder.eval()

    predicts = []
    gts = []

    context_size = 32 if config.tune_8k else 64
    num_samples = 256 if config.tune_8k else 512
    stft_layer = jit_model._model_8k.stft if config.tune_8k else jit_model._model.stft
    encoder_layer = jit_model._model_8k.encoder if config.tune_8k else jit_model._model.encoder

    with torch.no_grad():
        for _, (x, targets, masks) in tqdm(enumerate(loader), total=len(loader)):
            targets = targets.to(device)
            x = x.to(device)
            masks = masks.to(device)
            x = torch.nn.functional.pad(x, (context_size, 0))

            outs = []
            state = torch.zeros(0)
            for i in range(context_size, x.shape[1], num_samples):
                input_ = x[:, i-context_size:i+num_samples]
                out = stft_layer(input_)
                out = encoder_layer(out)
                out, state = decoder(out, state)
                outs.append(out)
            stacked = torch.cat(outs, dim=2).squeeze(1)

            predicts.extend(stacked[masks != 0].tolist())
            gts.extend(targets[masks != 0].tolist())

            loss = criterion(stacked, targets)
            loss = (loss * masks).mean()
            losses.update(loss.item(), masks.numel())
    score = roc_auc_score(gts, predicts)

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, round(score, 3)
