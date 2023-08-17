from IPython.display import Audio, display
import librosa as lb
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, title='Confusion Matrix', classes=[]):
    fig = plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.4f" % (c,), va='center', ha='center', size='xx-large')

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

    plt.show(block=False)
    return fig


def fade_in(waveform, sr):
    y = waveform.numpy()[0]

    onset_envs = lb.onset.onset_strength(y=y, sr=sr)
    onset_frame = list(onset_envs).index(max(onset_envs))
    if onset_frame > 3:
        onset_sample = lb.frames_to_samples(onset_frame - 3)
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


def plot_waveform(waveform, sr, title="Waveform"):
    if type(waveform).__name__ == 'Tensor':
        waveform = waveform.numpy()

    fig = plt.figure(figsize=(12, 8), dpi=100)

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    plt.plot(time_axis, waveform[0], linewidth=1)
    plt.grid(True)
    plt.xlim([0, time_axis[-1]])
    plt.title(title)
    plt.show(block=False)
    return fig


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig = plt.figure(figsize=(12, 8), dpi=100)

    if title is not None:
        plt.title(title)
    plt.ylabel(ylabel)
    plt.imshow(lb.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    plt.show(block=False)
    return fig


def plot_box(data, title=None, labels=None, ylabel=None):
    fig = plt.figure(figsize=(12, 8), dpi=100)

    bp = plt.boxplot(data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # fill with colors
    colors = ['#e5e2d6', '#c7bea0', '#dcc278', '#bf9a8e', '#829b7c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.grid(True)
    plt.show(block=False)
    return fig


def plot_violin(data, title=None, labels=None, ylabel=None):
    fig = plt.figure(figsize=(12, 8), dpi=100)

    bp = plt.violinplot(data,
                        showmeans=False,
                        showmedians=True)
                        # labels=labels)  # will be used to label x-ticks
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # fill with colors
    colors = ['#e5e2d6', '#c7bea0', '#dcc278', '#bf9a8e', '#829b7c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.grid(True)
    plt.show(block=False)
    return fig
