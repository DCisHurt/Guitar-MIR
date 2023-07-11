import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np


class GtFxDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 effect='all'):
        annotations = pd.read_csv(annotations_file)

        if effect == 'distortion':
            self.annotations = self._remove_item(annotations, 1)
        elif effect == 'chorus':
            self.annotations = self._remove_item(annotations, 2)
        elif effect == 'tremolo':
            self.annotations = self._remove_item(annotations, 3)
        elif effect == 'delay':
            self.annotations = self._remove_item(annotations, 4)
        elif effect == 'reverb':
            self.annotations = self._remove_item(annotations, 5)
        else:
            self.annotations = annotations

        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        labels = self._get_audio_sample_labels(index)
        value_classes = self._get_audio_sample_value_classes(index)
        parameters = self._get_audio_sample_parameters(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        filename = self.get_audio_sample_filename(index)
        return signal, labels, value_classes, parameters, filename

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 0] + ".wav")
        return path

    def _get_audio_sample_labels(self, index):
        labels = []
        for i in range(5):
            labels.append(np.float32(self.annotations.iloc[index, i + 1]))
        return labels

    def _get_audio_sample_value_classes(self, index):
        value_classes = []
        for i in range(5):
            value_classes.append(self.annotations.iloc[index, i + 6])
        return value_classes

    def _get_audio_sample_parameters(self, index):
        parameters = []
        for i in range(5):
            parameters.append(np.float32(self.annotations.iloc[index, i + 11]))
        return parameters

    def _remove_item(self, annotations, effect):
        remove_list = []
        for i in range(len(annotations)):
            if (annotations.iloc[i, effect] == 0):
                remove_list.append(i)
        return annotations.drop(remove_list, axis=0)

    def get_audio_sample_filename(self, index):
        return self.annotations.iloc[index, 0]
