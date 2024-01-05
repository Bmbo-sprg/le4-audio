import matplotlib.pyplot as plt
import numpy as np
import librosa
import pyaudio
from utils import extract_f0, frame2ceps, frame2vol, wave2cepsgram, wave2specgram, detect_speech, SpeechStatus

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

# learning phase
model = dict()
is_independent = input('Assume ceps are independent? (Y/n): ') != 'n'
for vowel in vowel_l:
  modeldata = modeldata_l[vowel]
  cepsgram = wave2cepsgram(modeldata, size_frame, size_shift, ceps_size=ceps_size)

  if is_independent:
    model[vowel] = (np.mean(cepsgram, axis=0), np.std(cepsgram, axis=0))
  else:
    model[vowel] = (np.mean(cepsgram), np.cov(cepsgram, rowvar=False))

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
  data = np.frombuffer(in_data, dtype=np.int16) / 32768.0

  speech_status = detect_speech(data, vol_threshold, zero_cross_threshold)
  ceps = frame2ceps(data * np.hamming(size_frame))

  if speech_status == SpeechStatus.QUIET or speech_status == SpeechStatus.UNVOICED:
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
  while stream.is_active():
    pass
  stream.stop_stream()
  stream.close()
  p.terminate()
else:
  filename = input('Filename of test data (without .wav): ')
  data = librosa.load(f"wave/{filename}.wav", sr=SR)[0]

  specgram = wave2specgram(data, size_frame, size_shift)
  recognized_vowel_l = []
  f0 = []
  for i in np.arange(0, len(data) - size_frame, size_shift):
    idx = int(i)
    frame = data[idx : idx + size_frame]
    ceps = frame2ceps(frame * np.hamming(size_frame))
    vol = frame2vol(frame)
    speech_status = detect_speech(frame, vol_threshold, zero_cross_threshold)

    f0.append(extract_f0(frame * np.hamming(size_frame), SR) if vol > vol_threshold else 0)

    score_d = recognize_vowel(model, vowel_l, ceps[:ceps_size])
    if speech_status == SpeechStatus.QUIET or speech_status == SpeechStatus.UNVOICED:
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
    np.flipud(np.array(specgram).T),
    extent=[0, len(data), 0, SR/2],
    aspect='auto',
    interpolation='nearest',
  )
  x_data = np.linspace(0, len(data), len(f0))
  ax1.plot(x_data, f0, color='blue', label='f0')
  ax1.set_yscale('log')
  ax1.set_ylim([60, SR / 2])

  ax2 = ax1.twinx()
  ax2.set_ylabel('Vowel')
  ax2.plot(x_data, recognized_vowel_l, color='red', label='vowel')
  fig.savefig(f'fig/{filename}_recognition.png')
  plt.clf()
