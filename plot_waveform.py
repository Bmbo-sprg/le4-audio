import matplotlib.pyplot as plt
import numpy as np
import librosa

SR = 16000

filename = input('Enter the filename (without .wav): ')
size_frame = int(input('Enter frame size (default: 512): ') or 512)
size_shift = int(input('Enter shift size (default: 160): ') or 160)
ceps_size = int(input('Enter cepstrum size (default: 13): ') or 13)

data, _ = librosa.load(f'wave/{filename}.wav', sr=SR)

fft = np.fft.rfft(data)
fft_log_abs_spec = np.log(np.abs(fft))
ceps = np.fft.rfft(fft_log_abs_spec)
spec_env = np.fft.irfft(ceps[:ceps_size], len(fft_log_abs_spec))

hamming_window = np.hamming(size_frame)
spectrogram = []
for i in np.arange(0, len(data) - size_frame, size_shift):
  idx = int(i)
  frame = data[idx : idx + size_frame] * hamming_window
  fft_spectro = np.fft.rfft(frame)
  fft_log_abs_spectro = np.log(np.abs(fft_spectro))
  spectrogram.append(fft_log_abs_spectro)

rms = []
vol = []
f_0 = []
zero_cross = []
for i in np.arange(0, len(data) - size_frame, size_shift):
  idx = int(i)
  frame = data[idx : idx + size_frame]
  rms_sample = np.sqrt(np.mean(frame**2))
  vol_sample = 20 * np.log10(rms_sample)

  autocorr = np.correlate(frame, frame, mode='full')
  autocorr = autocorr[len(autocorr)//2:]
  peakindices = [
    i for i in range(1, len(autocorr)-1) if autocorr[i-1] < autocorr[i] and autocorr[i+1] < autocorr[i]
  ]
  max_peak_index = max(peakindices, key=lambda i: autocorr[i])
  f_0_sample = SR / max_peak_index if vol_sample > -30 else 0  # -30 dB 以下は無音とみなす

  rms.append(rms_sample)
  vol.append(vol_sample)
  f_0.append(f_0_sample)

# waveform
fig = plt.figure(figsize=(10, 4))
plt.plot(data)
plt.xlabel('Sampling point')
fig.savefig(f'fig/{filename}_waveform.png')
plt.clf()

# spectrum
fig = plt.figure()
x_data = np.linspace((SR/2)/len(fft_log_abs_spec), SR/2, len(fft_log_abs_spec))
plt.plot(x_data, fft_log_abs_spec)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Log amplitude')
plt.xlim([0, SR/2])
fig.savefig(f'fig/{filename}_spectrum-whole.png')
plt.clf()

# spectrum (zoomed)
plt.plot(x_data, fft_log_abs_spec)
plt.xlim([0, 2000])
fig.savefig(f'fig/{filename}_spectrum-2000.png')
plt.clf()

# spectrum & envelope
plt.plot(x_data, fft_log_abs_spec)
plt.plot(x_data, spec_env)
fig.savefig(f'fig/{filename}_spectrum-envelope.png')
plt.clf()

# spectrogram
plt.xlabel('Sample')
plt.ylabel('Frequency [Hz]')
plt.imshow(
  np.flipud(np.array(spectrogram).T),
  extent=[0, len(data), 0, SR/2],
  aspect='auto',
  interpolation='nearest',
)
fig.savefig(f'fig/{filename}_spectrogram.png')
plt.clf()

# spectrogram (log)
plt.xlabel('Sample')
plt.ylabel('Frequency [Hz]')
plt.imshow(
  np.flipud(np.array(spectrogram).T),
  extent=[0, len(data), 0, SR/2],
  aspect='auto',
  interpolation='nearest',
)
plt.gca().set_yscale('log')
plt.ylim([60, 1000])
fig.savefig(f'fig/{filename}_spectrogram-log.png')
plt.clf()

# specgram-f0-log
plt.xlabel('Sample')
plt.ylabel('Frequency [Hz]')
plt.imshow(
  np.flipud(np.array(spectrogram).T),
  extent=[0, len(data), 0, SR/2],
  aspect='auto',
  interpolation='nearest',
)
x_data = np.linspace(0, len(data), len(f_0))
plt.plot(x_data, f_0, color='blue', label='f0')
plt.gca().set_yscale('log')
plt.ylim([60, 1000])
fig.savefig(f'fig/{filename}_spectrogram-f0-log.png')
plt.clf()

# rms
x_data = np.linspace(0, len(data), len(rms))
plt.xlabel('Sample')
plt.ylabel('RMS')
plt.plot(x_data, rms)
fig.savefig(f'fig/{filename}_rms.png')
plt.clf()

# volume
x_data = np.linspace(0, len(data), len(vol))
plt.xlabel('Sample')
plt.ylabel('Volume [dB]')
plt.plot(x_data, vol)
fig.savefig(f'fig/{filename}_volume.png')
plt.clf()
