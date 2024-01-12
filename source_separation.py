import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from utils import (
    wave2specgram,
)

sr = int(input('Enter sampling rate (default: 16000): ') or 16000)
size_frame = int(input('Enter frame size (default: 512): ') or 512)
size_shift = int(input('Enter shift size (default: 160): ') or 160)
k = int(input('Enter k (default: 2): ') or 2)
iter = int(input('Enter iteration (default: 100): ') or 100)

def nmf(specgram):
    n, m = specgram.shape
    h = np.random.rand(n, k)
    u = np.random.rand(k, m)
    for _ in range(iter):
        h = h * np.dot(specgram, u.T) / np.dot(u, np.dot(h, u).T).T
        u = u * np.dot(specgram.T, h).T / np.dot(h.T, np.dot(h, u))
    return h, u

def execute_file(filename):
    data = librosa.load(f"wave/{filename}.wav", sr=sr)[0]
    specgram = np.transpose(wave2specgram(data, size_frame, size_shift))
    h, u = nmf(np.abs(specgram))

    plt.rcParams['image.cmap'] = "inferno"
    fig = plt.figure(figsize=(20, 4))

    ax1 = fig.add_subplot(191)
    ax1.set_title('H')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.imshow(
        h,
        aspect='auto',
        origin='lower',
        interpolation='none',
        extent=[0, h.shape[1], 0, sr//2],
    )
    ax1.set_ylim([60, sr/2])

    ax2 = fig.add_subplot(1, 9, (2, 9))
    ax2.set_title('U')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Component')
    ax2.imshow(
        u,
        aspect='auto',
        origin='lower',
        interpolation='none',
        extent=[0, u.shape[1], 0, u.shape[0]],
    )

    fig.savefig(f'fig/{filename}_nmf.png')
    plt.clf()

    for i in range(k):
        specgram_sep = np.dot(h[:, i:i+1], u[i:i+1, :])
        data_sep = (np.fft.irfft(specgram_sep).flatten() * 32768.0).astype(np.int16)
        print(data_sep)
        scipy.io.wavfile.write(f'wave/{filename}_sep{i}.wav', sr, data_sep)

filename = input('Filename of test data (without .wav): ')
execute_file(filename)
