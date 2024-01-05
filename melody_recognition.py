import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from utils import (
    Note,
    SpeechStatus,
    spec2chroma,
    wave2specgram,
    detect_speech,
    extract_f0,
    frame2spec,
    hz2nn,
)

SR = int(input('Enter sampling rate (default: 16000): ') or 16000)
size_frame = int(input('Enter frame size (default: 512): ') or 512)
size_shift = int(input('Enter shift size (default: 160): ') or 160)
vol_threshold = int(input('Enter volume threshold (dB) (default: -30): ') or -30)
zero_cross_threshold = int(input('Enter zero-cross threshold (default: 70): ') or 70)
is_realtime = input('Realtime? (Y/n)') != 'n'

def extract_melody_autocorr(frame, sr):
    return hz2nn(extract_f0(frame, sr))

def extract_melody_shs(frame, sr):
    spec = frame2spec(frame)
    nn_max = sr // 2
    extended_chroma = np.zeros(hz2nn(nn_max) + 1)
    for i, s in enumerate(spec):
        if i == 0:
            continue
        nn = hz2nn(i * sr / 2 / len(spec))
        extended_chroma[nn] += np.abs(s)

    score_d = dict()
    for nn in range(56, 86):
        score_d[nn] = np.sum(extended_chroma[nn:nn_max:12])

    return max(score_d, key=lambda nn: score_d[nn])

def input_callback(in_data, frame_count, time_info, status_flags):
    wave = np.frombuffer(in_data, dtype=np.int16)
    wave = wave / 32768.0  # normalize

    status = detect_speech(wave, vol_threshold, zero_cross_threshold)
    if status == SpeechStatus.QUIET:
        return (in_data, pyaudio.paContinue)

    melody_autocorr = extract_melody_autocorr(wave, SR)
    melody_shs = extract_melody_shs(wave, SR)
    print(f'Note (autocorr): {Note(melody_autocorr % 12).name:2},{melody_autocorr:2}, '
          f'Note (shs): {Note(melody_shs % 12).name:2},{melody_shs:2}')

    return (in_data, pyaudio.paContinue)

def execute_realtime():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SR,
        input=True,
        frames_per_buffer=size_frame,
        stream_callback=input_callback,
    )
    stream.start_stream()
    while stream.is_active():
        pass
    stream.stop_stream()
    stream.close()
    p.terminate()

def execute_file(filename):
    data = librosa.load(f"wave/{filename}.wav", sr=SR)[0]
    specgram = wave2specgram(data, size_frame, size_shift)
    log_specgram = np.log(np.abs(specgram))
    melody_autocorr_l = []
    melody_shs_l = []
    for i in np.arange(0, len(data) - size_frame, size_shift):
        idx = int(i)
        frame = data[idx : idx + size_frame]
        melody_autocorr_l.append(extract_melody_autocorr(frame, SR))
        melody_shs_l.append(extract_melody_shs(frame, SR))

    plt.rcParams['image.cmap'] = "inferno"
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.set_title('waveform')
    ax1.plot(data)
    ax2 = fig.add_subplot(212)
    ax2.set_title('spectrogram & melody')
    ax2.imshow(
        np.flipud(np.array(log_specgram).T),
        extent=[0, len(log_specgram), 0, SR/2],
        aspect='auto',
        interpolation='nearest',
    )
    ax2.set_yscale('log')
    ax2.set_ylim([60, SR / 2])
    ax3 = ax2.twinx()
    ax3.plot(melody_autocorr_l, label='autocorr', color='red')
    ax3.plot(melody_shs_l, label='shs', color='blue')
    ax3.set_ylim([56, 86])

    fig.savefig(f'fig/{filename}_melody.png')
    plt.clf()

if is_realtime:
    execute_realtime()
else:
    filename = input('Filename of test data (without .wav): ')
    execute_file(filename)
