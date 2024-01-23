from enum import Enum
import math
import numpy as np


class SpeechStatus(Enum):
    QUIET = 0
    UNVOICED = 1
    VOICED = 2


class Note(Enum):
    C = 0
    CS = 1
    D = 2
    DS = 3
    E = 4
    F = 5
    FS = 6
    G = 7
    GS = 8
    A = 9
    AS = 10
    B = 11


CHORDS_L = [
    # name, root, third, fifth
    ['C Major', Note.C, Note.E, Note.G],
    ['C Minor', Note.C, Note.DS, Note.G],
    ['C# Major', Note.CS, Note.F, Note.GS],
    ['C# Minor', Note.CS, Note.E, Note.GS],
    ['D Major', Note.D, Note.FS, Note.A],
    ['D Minor', Note.D, Note.F, Note.A],
    ['D# Major', Note.DS, Note.G, Note.AS],
    ['D# Minor', Note.DS, Note.FS, Note.AS],
    ['E Major', Note.E, Note.GS, Note.B],
    ['E Minor', Note.E, Note.G, Note.B],
    ['F Major', Note.F, Note.A, Note.C],
    ['F Minor', Note.F, Note.GS, Note.C],
    ['F# Major', Note.FS, Note.AS, Note.CS],
    ['F# Minor', Note.FS, Note.A, Note.CS],
    ['G Major', Note.G, Note.B, Note.D],
    ['G Minor', Note.G, Note.AS, Note.D],
    ['G# Major', Note.GS, Note.C, Note.DS],
    ['G# Minor', Note.GS, Note.B, Note.DS],
    ['A Major', Note.A, Note.CS, Note.E],
    ['A Minor', Note.A, Note.C, Note.E],
    ['A# Major', Note.AS, Note.D, Note.F],
    ['A# Minor', Note.AS, Note.CS, Note.F],
    ['B Major', Note.B, Note.DS, Note.FS],
    ['B Minor', Note.B, Note.D, Note.FS],
]


def nn2hz(nn):
    return 440 * 2 ** ((nn - 69) / 12)


def hz2nn(hz):
    return int(round(12 * math.log2(hz / 440) + 69))


def spec2chroma(spec, sr):
    chroma = np.zeros(12)
    for i, s in enumerate(spec):
        if i == 0:
            continue
        nn = hz2nn(i * sr / 2 / len(spec))
        chroma[nn % 12] += np.abs(s)
    return chroma


def wave2specgram(wave, size_frame, size_shift):
    specgram = []
    for i in np.arange(0, len(wave) - size_frame, size_shift):
        idx = int(i)
        frame = wave[idx:idx + size_frame] * np.hamming(size_frame)
        spec = np.fft.rfft(frame)
        specgram.append(spec)
    return specgram


def wave2cepsgram(wave, size_frame, size_shift, ceps_size=None):
    cepsgram = []
    for i in np.arange(0, len(wave) - size_frame, size_shift):
        idx = int(i)
        frame = wave[idx:idx + size_frame] * np.hamming(size_frame)
        spec = np.log(np.abs(np.fft.rfft(frame)))
        ceps = np.real(np.fft.rfft(spec))
        cepsgram.append(ceps if ceps_size is None else ceps[:ceps_size])
    return cepsgram


def frame2spec(frame):
    return np.fft.rfft(frame * np.hamming(len(frame)))


def frame2ceps(frame, ceps_size=None):
    spec = np.log(np.abs(np.fft.rfft(frame)))
    ceps = np.real(np.fft.rfft(spec))
    return ceps if ceps_size is None else ceps[:ceps_size]


def frame2vol(frame):
    return 20 * np.log10(np.sqrt(np.mean(frame**2)))


def detect_speech(frame, vol_threshold, zero_cross_threshold):
    vol = frame2vol(frame)
    zero_cross = int(sum(np.abs(np.diff(np.sign(frame))) // 2))

    if vol < vol_threshold:
        return SpeechStatus.QUIET
    if zero_cross > zero_cross_threshold:
        return SpeechStatus.UNVOICED
    return SpeechStatus.VOICED


def extract_f0(frame, sr, min_f0=60, max_f0=1000):
    fft = np.fft.fft(frame)
    autocorr = np.real(np.fft.ifft(fft * np.conj(fft)))
    peakindices = [
        i for i in range(sr // max_f0, sr // min_f0)
        if autocorr[i-1] < autocorr[i] and autocorr[i+1] < autocorr[i]
    ]
    if len(peakindices) == 0:
        max_peak_index = np.argmax(autocorr[sr // max_f0:sr // min_f0])
    else:
        max_peak_index = max(peakindices, key=lambda i: autocorr[i])
    return sr / max_peak_index
