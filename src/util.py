from IPython.display import Audio, display
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion Matrix', classes=[]):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), va='center', ha='center', size='xx-large')

    plt.imshow(cm, cmap=plt.cm.YlOrRd, alpha=0.5)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    # plt.savefig(savename, format='png')
    plt.show()

def fade_in(waveform, sr):
    y = waveform.numpy()[0]

    onset_envs = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frame = list(onset_envs).index(max(onset_envs))
    if onset_frame > 3:
        onset_sample = librosa.frames_to_samples(onset_frame - 3)
        end = onset_sample
        fade_curve = np.linspace(0.0, 0.25, end)

        for nChannels, _ in enumerate(waveform):
            waveform[nChannels][0:end] = waveform[nChannels][0:end] * fade_curve

    return waveform


def right_pad(waveform, sr, length):
    num_missing_samples = length * sr - len(waveform[0])
    last_dim_padding = (0, num_missing_samples)
    waveform = torch.nn.functional.pad(waveform, last_dim_padding)

    return waveform


def play_audio(waveform, sample_rate):
    if type(waveform).__name__ == 'Tensor':
        waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")
