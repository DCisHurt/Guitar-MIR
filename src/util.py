from IPython.display import Audio, display
import librosa as lb
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from fastdtw import fastdtw


def plot_confusion_matrix(cm, title='Confusion Matrix', classes=[]):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams.update({'font.size': 14})
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


def plot_violin(data, title=None, labels=None, ylabel=None, outlier=True):
    fig = plt.figure(figsize=(12, 8), dpi=200)
    plt.rcParams.update({'font.size': 12})

    bp = plt.violinplot(data,
                        showmeans=False,
                        showmedians=True,
                        showextrema=outlier)
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # fill with colors
    colors = ['#e5e2d6', '#c7bea0', '#dcc278', '#bf9a8e', '#829b7c']
    for patch, color in zip(bp['bodies'], colors):
        # patch.set_facecolor(color)
        patch.set_edgecolor('000000')
        patch.set_alpha(0.5)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.grid(True)

    plt.show(block=False)
    return fig


def load_train_data(train_dir=None):
    if train_dir is None:
        train_dir = '_log/Legacy/Training/'

    train_acc = np.genfromtxt(train_dir + 'Accuracy_train.csv', delimiter=',')
    train_loss = np.genfromtxt(train_dir + 'Loss_train.csv', delimiter=',')
    test_acc = np.genfromtxt(train_dir + 'Accuracy_test.csv', delimiter=',')
    test_loss = np.genfromtxt(train_dir + 'Loss_test.csv', delimiter=',')

    train_acc = np.delete(train_acc, 0, 0)
    train_acc = np.delete(train_acc, 0, 1)
    train_acc = np.rot90(train_acc, 1)
    test_acc = np.delete(test_acc, 0, 0)
    test_acc = np.delete(test_acc, 0, 1)
    test_acc = np.rot90(test_acc, 1)

    train_loss = np.delete(train_loss, 0, 0)
    train_loss = np.delete(train_loss, 0, 1)
    train_loss = np.rot90(train_loss, 1)
    test_loss = np.delete(test_loss, 0, 0)
    test_loss = np.delete(test_loss, 0, 1)
    test_loss = np.rot90(test_loss, 1)

    acc = [train_acc, test_acc]
    loss = [train_loss, test_loss]

    return acc, loss


def load_train_data_error(train_dir=None):
    if train_dir is None:
        train_dir = '_log/Legacy/Training/'

    train_error = np.genfromtxt(train_dir + 'Error_train.csv', delimiter=',')
    train_loss = np.genfromtxt(train_dir + 'Loss_train.csv', delimiter=',')
    test_error = np.genfromtxt(train_dir + 'Error_test.csv', delimiter=',')
    test_loss = np.genfromtxt(train_dir + 'Loss_test.csv', delimiter=',')

    train_error = np.delete(train_error, 0, 0)
    train_error = np.delete(train_error, 0, 1)
    train_error = np.rot90(train_error, 1)
    test_error = np.delete(test_error, 0, 0)
    test_error = np.delete(test_error, 0, 1)
    test_error = np.rot90(test_error, 1)

    train_loss = np.delete(train_loss, 0, 0)
    train_loss = np.delete(train_loss, 0, 1)
    train_loss = np.rot90(train_loss, 1)
    test_loss = np.delete(test_loss, 0, 0)
    test_loss = np.delete(test_loss, 0, 1)
    test_loss = np.rot90(test_loss, 1)

    error = [train_error, test_error]
    loss = [train_loss, test_loss]

    return error, loss


def plot_train_line(data, num_subplot, subtitle, legend_loc,
                    x_label, y_label, x_ticks, y_ticks,
                    hight, width, title=None):

    if len(data) != num_subplot and num_subplot != 1:
        print('Error: data must have {} elements'.format(num_subplot))
        return None

    fig = plt.figure(figsize=(width, hight), dpi=200)
    plt.rcParams.update({'font.size': 14})

    if num_subplot == 1:
        plt.plot(data[0][1], data[0][0], label='Train', linewidth=2, marker='s', markersize=10)
        plt.plot(data[1][1], data[1][0], label='Test', linewidth=2, marker='s', markersize=10)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.legend(loc=legend_loc)
        plt.title(subtitle)
    else:
        for i in range(num_subplot):
            plt.subplot(1, num_subplot, i+1)
            plt.plot(data[i][0][1], data[i][0][0], label='Train',
                     linewidth=2, marker='s', markersize=10)
            plt.plot(data[i][1][1], data[i][1][0], label='Test',
                     linewidth=2, marker='s', markersize=10)
            plt.xlabel(x_label)
            if i == 0:
                plt.ylabel(y_label)
            if title is not None:
                plt.title(title[i])
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
            plt.legend(loc=legend_loc)
        plt.suptitle(subtitle)
    plt.show(block=False)
    return fig


def cm_score(data):
    return np.round([accuracy_score(data[0], data[1]),
                    precision_score(data[0], data[1]),
                    recall_score(data[0], data[1]),
                    f1_score(data[0], data[1])], 4)


def multi_bar_plot(*data, N, color_list, xlabels, ylabel, title, legend):
    ind = np.arange(N)*3 + .35
    width = 0.35
    xtra_space = 0.1

    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams.update({'font.size': 12})

    for i in range(len(data)):
        plt.bar(ind + (width + xtra_space)*i, data[i], width, color=color_list[i])

    # add some text for labels, title and axes ticks
    # plt.set_ylabel('Population, millions')
    # plt.set_title('Population: Age Structure')
    plt.legend(legend)
    plt.ylabel(ylabel)
    plt.xticks(ind+(width+xtra_space)*2, xlabels)
    plt.title(title)

    plt.show(block=False)
    return fig


def hist_plot(data, bins, xlabel, ylabel, title):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams.update({'font.size': 12})

    plt.hist(data, histtype='stepfilled', label="values", bins=bins)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=bins)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show(block=False)
    return fig


def mfcc_dist(a, b):
    dist = 0
    for x, y in zip(a, b):
        dist = dist + (x - y) * (x - y)
    return np.sqrt(dist)


def similarity_percentage(waveform, remix):
    distance, _ = fastdtw(waveform, remix, dist=mfcc_dist)
    max_possible_distance = 64 * 64
    normalized_distance = distance / max_possible_distance
    similarity_percentage = (1 - normalized_distance) * 100
    return similarity_percentage
