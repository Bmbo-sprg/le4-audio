import librosa
import numpy as np
import scipy.io.wavfile

def generate_sinusoid(sr, freq, duration):
    sampling_interval = 1.0 / sr
    t = np.arange(sr * duration) * sampling_interval
    wave = np.sin(2.0 * np.pi * freq * t)
    return wave

sr = int(input('Enter sampling rate (default: 16000): ') or 16000)

filename = input('Filename of test data (without .wav): ')
data = librosa.load(f"wave/{filename}.wav", sr=sr)[0]

sin_wave = generate_sinusoid(sr, 10.0, len(data) / sr) * 0.2 + 1.0

data = data[:min(len(data), len(sin_wave))]
sin_wave = sin_wave[:min(len(data), len(sin_wave))]

data = data * sin_wave

data = (data * 32768.0).astype(np.int16)
scipy.io.wavfile.write(f'wave/{filename}_sin.wav', sr, data)
