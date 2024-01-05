import matplotlib.pyplot as plt
import numpy as np
import librosa
import pyaudio
import time
from collections import deque

SR = 16000

vowel_l = ['a', 'i', 'u', 'e', 'o']
modeldata_l = dict()
for vowel in vowel_l:
  modeldata = librosa.load(f'wave/{input(f"Filename of model data /{vowel}/ (without .wav): ")}.wav', sr=SR)[0]
  modeldata_l[vowel] = modeldata

size_frame = int(input('Enter frame size (default: 512): ') or 512)
size_shift = int(input('Enter shift size (default: 160): ') or 160)
ceps_size = int(input('Enter cepstrum size (default: 13): ') or 13)
vol_threshold = int(input('Enter volume threshold (dB) (default: -30): ') or -30)
zero_cross_threshold = int(input('Enter zero-cross threshold (default: 70): ') or 70)

hamming_window = np.hamming(size_frame)

# learning phase
model = dict()
is_independent = input('Assume ceps are independent? (Y/n): ') != 'n'
for vowel in vowel_l:
  modeldata = modeldata_l[vowel]
  ceps_l = []
  for i in np.arange(0, len(modeldata) - size_frame, size_shift):
    idx = int(i)
    frame = modeldata[idx : idx + size_frame] * hamming_window
    ceps = np.real(np.fft.rfft(np.log(np.abs(np.fft.rfft(frame)))))
    ceps_l.append(ceps[:ceps_size])

  if is_independent:
    model[vowel] = (np.mean(ceps_l, axis=0), np.std(ceps_l, axis=0))
  else:
    model[vowel] = (np.mean(ceps_l), np.cov(ceps_l, rowvar=False))

# test phase
def recognize_vowel(model, vowel_l, ceps):
  score_d = dict()
  ceps = ceps[:ceps_size]
  for vowel in vowel_l:
    if is_independent:
      mean, std = model[vowel]
      score_d[vowel] = -np.sum((ceps - mean)**2 / (2 * std**2) + np.log(std))
    else:
      mean, cov = model[vowel]
      score_d[vowel] = -np.sum(
        np.transpose(ceps - mean) @ np.linalg.inv(cov) @ (ceps - mean) / 2 + np.log(np.linalg.det(cov))
      )
  return score_d

def input_callback(in_data, frame_count, time_info, status_flags):
  data = np.frombuffer(in_data, dtype=np.int16)
  data = data / 32768.0
  data_hammed = data * hamming_window
  vol = 20 * np.log10(np.sqrt(np.mean(data**2)))
  zero_cross = int(sum(np.abs(np.diff(np.sign(data))) // 2))

  ceps = np.real(np.fft.rfft(np.log(np.abs(np.fft.rfft(data_hammed)))))

  if vol < vol_threshold or zero_cross > zero_cross_threshold:
    # not speaking, or unvoiced
    return (in_data, pyaudio.paContinue)

  score_d = recognize_vowel(model, vowel_l, ceps)
  vowel_recognized = max(score_d, key=lambda k: score_d[k])
  print(f'{vowel_recognized} (a: {score_d["a"]:.2f}, i: {score_d["i"]:.2f}, u: {score_d["u"]:.2f}, e: {score_d["e"]:.2f}, o: {score_d["o"]:.2f})')
  return (in_data, pyaudio.paContinue)

if input('Realtime? (Y/n): ') != 'n':
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
  time.sleep(100)
  stream.stop_stream()
  exit()
else:
  filename = input('Filename of test data (without .wav): ')
  data = librosa.load(f"wave/{filename}.wav", sr=SR)[0]
  hamming_window = np.hamming(size_frame)
  spectrogram = []
  recognized_vowel_l = []
  f_0 = []
  for i in np.arange(0, len(data) - size_frame, size_shift):
    idx = int(i)
    frame = data[idx : idx + size_frame] * hamming_window
    frame_hammed = data[idx : idx + size_frame] * hamming_window
    spec = np.log(np.abs(np.fft.rfft(frame_hammed)))
    ceps = np.real(np.fft.rfft(spec))
    vol = 20 * np.log10(np.sqrt(np.mean(frame**2)))
    zero_cross = int(sum(np.abs(np.diff(np.sign(frame))) // 2))
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peakindices = [
      i for i in range(1, len(autocorr)-1) if autocorr[i-1] < autocorr[i] and autocorr[i+1] < autocorr[i]
    ]
    max_peak_index = max(peakindices, key=lambda i: autocorr[i])
    f_0.append(SR / max_peak_index if vol > vol_threshold else 0)
    spectrogram.append(spec)
    score_d = recognize_vowel(model, vowel_l, ceps[:ceps_size])
    print(vol)
    if vol < vol_threshold or zero_cross > zero_cross_threshold:
      recognized_vowel_l.append(0)
    else:
      recognized_vowel = max(score_d, key=lambda k: score_d[k])
      recognized_vowel_l.append(vowel_l.index(recognized_vowel) + 1)

  # plot
  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ax1.set_xlabel('Sample')
  ax1.set_ylabel('Frequency [Hz]')
  ax1.imshow(
    np.flipud(np.array(spectrogram).T),
    extent=[0, len(data), 0, SR/2],
    aspect='auto',
    interpolation='nearest',
  )
  x_data = np.linspace(0, len(data), len(f_0))
  ax1.plot(x_data, f_0, color='blue', label='f0')
  ax1.set_yscale('log')
  ax1.set_ylim([60, SR / 2])

  ax2 = ax1.twinx()
  ax2.set_ylabel('Vowel')
  ax2.plot(x_data, recognized_vowel_l, color='red', label='vowel')
  fig.savefig(f'fig/{filename}_recognition.png')
  plt.clf()
